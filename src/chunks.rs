use std::{collections::VecDeque, f32::consts::TAU};

use bevy::{
    asset::RenderAssetUsages,
    ecs::entity_disabling::Disabled,
    input::common_conditions::input_just_pressed,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    platform::collections::HashSet,
    prelude::*,
};
use glam::{Mat2, Mat3A, Vec2, Vec3A};
use hexasphere::shapes::IcoSphere;
use itertools::Itertools;
use num::{One, ToPrimitive, Zero};
use rand::seq::IteratorRandom;

use crate::{assets::AssetHandles, triangle::cartesian_to_barycentric};

// ========================================================
// Trig & maths
// ========================================================

/// Area of a triangle given the 3 points
fn area2(triangle: [Vec3A; 3]) -> f32 {
    assert!((1. - triangle[0].length()).abs() < EPS);
    assert!((1. - triangle[1].length()).abs() < EPS);
    assert!((1. - triangle[2].length()).abs() < EPS);

    let base_mid = triangle[0].midpoint(triangle[1]);
    let height2 = (triangle[2] - base_mid).length_squared();
    let width2 = (triangle[1] - triangle[0]).length_squared();

    0.25 * height2 * width2
}

const EPS: f32 = 1e-3;

/// https://en.wikipedia.org/wiki/Triple_product#Properties
fn coplanar(v0: Vec3A, v1: Vec3A, point: Vec3A) -> bool {
    v0.cross(v1).dot(point).abs() < EPS
}

fn colinear(v: Vec3A, point: Vec3A) -> bool {
    (v.dot(point) - v.length() * point.length()).abs() < EPS
}

fn obtuse(v: Vec3A, point: Vec3A) -> bool {
    v.dot(point) < 0.
}

/// Project point towards the origin onto the line formed by v0-v1.
/// v0, v1, point must be coplanar
fn project(v0: Vec3A, v1: Vec3A, point: Vec3A) -> Vec3A {
    let d = v1 - v0; // direction of the line

    // Solve: s * P = v0 + t*d  =>  s * P - t*d = v0
    // Use cross products to isolate t:
    let p_cross_d = point.cross(d);
    let denom = p_cross_d.dot(p_cross_d); // |P × d|²

    let v0_cross_d = v0.cross(d);
    let s = v0_cross_d.dot(p_cross_d) / denom;

    s * point
}

/// Cosine similarity: (a . b) / |a||b|
fn cossim(v0: Vec3A, v1: Vec3A) -> f32 {
    v0.dot(v1) / (v0.length_squared() * v1.length_squared()).sqrt()
}

/// Absolute angle between two vectors
fn arc_distance(v0: Vec3A, v1: Vec3A) -> f32 {
    cossim(v0, v1).acos()
}

/// https://en.wikipedia.org/wiki/Heron%27s_formula
fn heron_area(a: f32, b: f32, c: f32) -> f32 {
    let s = (a + b + c) / 2.;

    (s * (s - a) * (s - b) * (s - c)).sqrt()
}

pub fn almost_equal(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

// ========================================================
// ECS bits
// ========================================================

#[derive(Resource)]
pub struct WorldRoot {
    /// Each chunk is one of the base Icosohedron faces
    /// 0-5: Top pentagon
    /// 5-10: Top ring
    /// 10-15: Bottom ring
    /// 15-20: Bottom pentagon
    root_chunks: [Entity; 20],

    /// First index is from
    /// inner indices are to
    /// Index into above chunk array
    siblings: [[usize; 3]; 20],
}

impl WorldRoot {
    /// Gets the singlings for a specific base chunk
    pub fn get_siblings(&self, chunk: Entity) -> Option<[Entity; 3]> {
        let index = self.root_chunks.iter().position(|e| *e == chunk)?;
        let siblings = self.siblings[index].map(|i| self.root_chunks[i]);
        Some(siblings)
    }
}

#[derive(Component)]
pub struct ChunkPos(pub Vec3A);

#[derive(Debug, Clone, Copy)]
enum TriangleCmp {
    Outside,
    /// Index of triangle vertex
    Corner(usize),
    Edge {
        v0: usize,
        v1: usize,
        /// How far along V0-V1
        t: f32,
    },
    Inside,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Triangle(pub [Vec3A; 3]);

impl Triangle {
    fn area(&self) -> f32 {
        area2(self.0).sqrt()
    }

    fn edges(&self) -> [(Vec3A, Vec3A); 3] {
        self.0
            .iter()
            .circular_tuple_windows::<(_, _)>()
            .map(|(v0, v1)| (*v0, *v1))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Centre point on unit circle
    pub fn centre(&self) -> Vec3A {
        (self.0.iter().sum::<Vec3A>() / 3.).normalize()
    }

    /// Distance between centre point and edge midpoint in radians
    fn edge_arc_radius(&self) -> f32 {
        let edge_midpoint = self.0[0].midpoint(self.0[1]);

        arc_distance(self.centre(), edge_midpoint)
    }

    fn corner_arc_radius(&self) -> f32 {
        arc_distance(self.centre(), self.0[0])
    }

    fn cmp_bary(&self, point: Vec3A, subdivisions: u32) -> TriangleCmp {
        let bary = cartesian_to_barycentric(self.0, point);
        if bary.length < 0. {
            // Other side of world
            return TriangleCmp::Outside;
        }

        let Some(bary) = bary.snap_even(subdivisions) else {
            return TriangleCmp::Outside;
        };

        if let Some(i) = bary.distances.iter().position(|d| d.is_one()) {
            // Corner oposite this edge
            return TriangleCmp::Corner((i + 2) % 3);
        }

        if let Some(i) = bary.distances.iter().position(|d| d.is_zero()) {
            let i0 = i;
            let i1 = (i + 1) % 3;
            // Along this edge
            // TODO: check this index
            let t = bary.distances[(i + 2) % 3];
            return TriangleCmp::Edge {
                v0: i0,
                v1: i1,
                t: t.to_f32().unwrap(),
            };
        }

        // Inside
        TriangleCmp::Inside
    }

    fn subdivide(&self) -> [Self; 4] {
        // Create midpoints of edges on unit sphere
        let midpoints = self.edges().map(|(v0, v1)| v0.midpoint(v1).normalize());

        // Create outer triangles
        let outer = midpoints
            .iter()
            .circular_tuple_windows()
            .zip(self.0.iter().cycle().skip(1))
            // CCW order
            .map(|((v0, v1), v2)| [*v2, *v1, *v0]);

        // Inner triangle is just between the new midpoints
        let inner = std::iter::once(midpoints);

        // Wrap in triangle
        outer
            .chain(inner)
            .map(Triangle)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    // Gets a basic mesh of a single triangle for this chunk
    pub fn get_mesh(&self) -> Mesh {
        let indices = Indices::U32(vec![0, 1, 2]);
        let positions = self.0.iter().map(|t| t.to_array()).collect::<Vec<_>>();

        // Set UV to point at the orientation of the triangle
        let uvs = (0..3)
            .map(|i| {
                let angle = TAU * i as f32 / 3.;
                let (y, x) = angle.sin_cos();
                [y, x].map(|x| x / 2. + 0.5)
            })
            .collect::<Vec<_>>();

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(indices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_computed_normals()
    }
}

#[derive(Component, Debug)]
pub struct AccTriangle(pub [Vec3A; 3]);

impl AccTriangle {
    /// Conversion for all of the methods
    pub fn as_triangle(&self) -> Triangle {
        Triangle(self.0)
    }
}

/// Level of subdivision this triangle is part of. Root == 0
#[derive(Component, Debug)]
pub struct SubdivisionLevel(usize);

#[derive(Bundle)]
pub struct ChunkBundle {
    pub pos: ChunkPos,
    pub triangle: Triangle,
    pub mesh: Mesh3d,
    pub transform: Transform,
    pub material: MeshMaterial3d<StandardMaterial>,
    pub acc_triangle: AccTriangle,
    pub subdivision: SubdivisionLevel,
}

#[derive(Component)]
#[relationship(relationship_target = ChildrenChunks)]
pub struct ParentChunk(Entity);

#[derive(Component)]
#[relationship_target(relationship = ParentChunk)]
pub struct ChildrenChunks(Vec<Entity>);

#[derive(EntityEvent)]
pub struct SubdivideChunk(Entity);

pub fn subdivide_chunk(
    event: On<SubdivideChunk>,
    mut commands: Commands,
    chunks: Query<(
        &Triangle,
        &MeshMaterial3d<StandardMaterial>,
        &SubdivisionLevel,
    )>,
    mut meshes: ResMut<Assets<Mesh>>,
) -> Result {
    let (triangle, material, level) = chunks.get(event.0)?;
    let new_triangles = triangle.subdivide();

    let new_bundles = new_triangles.map(|t| {
        let mesh = meshes.add(t.get_mesh());
        ChunkBundle {
            pos: ChunkPos(t.centre()),
            triangle: t,
            mesh: Mesh3d(mesh),
            transform: Transform::IDENTITY,
            material: material.clone(),
            acc_triangle: AccTriangle(t.0),
            subdivision: SubdivisionLevel(level.0 + 1),
        }
    });

    // Spawn chunk x4
    let children = new_bundles.map(|b| commands.spawn(b).id());

    // Add debug text
    children.iter().zip(new_triangles).for_each(|(e, t)| {
        commands.spawn((
            Text2d::new(format!("{:?}", e)),
            Transform::from_translation((t.centre()).to_array().into()),
            TextColor::WHITE,
        ));
    });

    // Add children to this chunk
    commands
        .entity(event.0)
        .add_related::<ParentChunk>(&children)
        // Make the parent invisible
        .insert(Disabled);

    Ok(())
}

fn init_world(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, assets: Res<AssetHandles>) {
    // Base sphere
    let sphere = IcoSphere::new(0, |_| ());
    let vertices = sphere.raw_points();

    // Create chunk triangles
    let triangles = sphere
        .get_all_indices()
        .into_iter()
        .array_chunks::<3>()
        .map(|indices| Triangle(indices.map(|i| vertices[i as usize])));

    // Get adjacency LUT
    let indices = sphere
        .get_all_indices()
        .into_iter()
        .array_chunks::<3>()
        .map(HashSet::from)
        .collect::<Vec<_>>();

    let mut siblings = [[0; 3]; 20];
    for (i, from) in indices.iter().enumerate() {
        let mut offset = 0;
        for (j, to) in indices.iter().enumerate() {
            if i == j {
                continue;
            }

            if from.intersection(to).count() == 2 {
                // Shared edge, therefore siblings
                siblings[i][offset] = j;
                offset += 1;
            }
        }
    }

    // Spawn shape
    let chunks = triangles
        .map(|triangle| {
            let mesh = triangle.get_mesh();
            let mesh = meshes.add(mesh);

            let pos = triangle.centre();

            commands
                .spawn(ChunkBundle {
                    pos: ChunkPos(pos),
                    triangle,
                    mesh: Mesh3d(mesh),
                    transform: Transform::IDENTITY,
                    material: MeshMaterial3d(assets.hue_material.clone()),
                    acc_triangle: AccTriangle(triangle.0),
                    subdivision: SubdivisionLevel(0),
                })
                .id()
        })
        .collect_array()
        .expect("Should be exactly 20 faces");

    commands.insert_resource(WorldRoot {
        root_chunks: chunks,
        siblings,
    });
}

/// Iterate over all non-disabled chunks which share an edge with this one
/// Sorted from low to high resolution
fn iter_adjacent(
    entity: Entity,
    world: &WorldRoot,
    relationships: Query<(
        &Triangle,
        &SubdivisionLevel,
        Option<&ChildrenChunks>,
        Has<Disabled>,
    )>,
) -> Vec<Entity> {
    let mut queue = VecDeque::from(world.root_chunks);

    let (test_triangle, test_level, _, _) = relationships.get(entity).unwrap();

    let mut chunks = vec![];
    while let Some(candidate) = queue.pop_front() {
        let (triangle, level, children, disabled) = relationships.get(candidate).unwrap();
        let cmp = test_triangle
            .0
            .map(|v| triangle.cmp_bary(v, test_level.0.max(level.0) as u32));

        let mut cmp_counts = [0_usize; 4];
        for c in cmp {
            let i = match c {
                TriangleCmp::Outside => 0,
                TriangleCmp::Corner(_) => 1,
                TriangleCmp::Edge { .. } => 2,
                TriangleCmp::Inside => 3,
            };
            cmp_counts[i] += 1;
        }

        println!("{:?} -> {:?}: {:?}", entity, candidate, cmp_counts);

        match cmp_counts {
            // [O C E I]
            // Ancestor fully inside, inside touching edge
            [_, _, _, 1..] |
            // Parent centre
            [_, _, 3.., _] |
            // Ancestor corner
            [_, 1, 2, _] => {
                assert!(
                    disabled,
                    "Ancestor chunks should be disabled {:?}",
                    (entity, candidate)
                );
                let children = children.expect("Ancestor chunks must have children");
                queue.extend(children.0.iter().copied());

            }

            // Direct sibling
            [1, 2, _, _] |
            // (gr)unc corner-edge
            [1, 1, 1, _] |
            // (gr)unc edge-edge
            [1, _, 2, _] |
            // Corner on edge but not adjacent
            [2, _, 1, _] => {
                // An edge is shared with this triangle
                if disabled {
                    // Adjacent disabled chunk
                    let children = children.expect("Ancestor chunks must have children");
                    queue.extend(children.0.iter().copied());
                } else {
                    // Adjacent enabled Leaf chunk
                    assert!(children.is_none(), "Enabled chunks should be leaves");
                    chunks.push(candidate);
                }
            }


            [_, 3.., _, _] => {
                // Self
            }
            [3.., _, _, _] => {
                // Fully outside
            }
            [2, 1, _, _] => {
                // Shared corner but not adjacent
            }
            x => unreachable!("Invalid combination: {:?}", x),
        }
    }

    chunks
}

// TODO: I need a more robust way of figuring this out. Really I should be using the chunk id/LOD
// layer, traversing up the tetra-tree appropriately and reading out the vertices directly rather
// than trying to do collision checks & match vertices. Floating errors are a bitch.
//
//  Algorithm:
//      Given a chunk
//      - Assume all lower LOD chunks have been processed and are at the correct height
//      For each vertex:
//      - Check each parent's siblings
//      - If invisible, skip
//      - If our vertex is within the radius of the sibling, we need to adjust this vertex
//      - Find the two vertices of the sibling which are closest to us, and take the midpoint.
//
//      Base chunks don't need to be adjusted
//      Special case handling middle triangle. Take vertices from siblings, since middle will always
//      be processed last
//
fn adjust_mesh_height(
    _commands: Commands,
    world: Res<WorldRoot>,
    mesh_handles: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut acc_triangles: Query<&mut AccTriangle>,
    chunks: Query<(
        &Triangle,
        &SubdivisionLevel,
        Option<&ChildrenChunks>,
        Option<&ParentChunk>,
        Has<Disabled>,
    )>,
    relationships: Query<(
        &Triangle,
        &SubdivisionLevel,
        Option<&ChildrenChunks>,
        Has<Disabled>,
    )>,
) -> Result {
    let mut queue = VecDeque::new();
    // Add all base+1 chunks to the queue, base chunks don't need processing
    for entity in world.root_chunks {
        let (_, _, children, _, _) = chunks.get(entity)?;
        if let Some(children) = children {
            queue.extend(children.0.clone());
        }
    }

    // Traverse in breadth-first manner
    while let Some(entity) = queue.pop_front() {
        let (triangle, level, children, _, disabled) = chunks.get(entity)?;

        if disabled && let Some(children) = children {
            // This one is disabled, so it doesn't need adjusting. Adjust children instead
            queue.extend(children.0.iter().copied());
            continue;
        }
        println!("=========================================================");
        println!("adjusting {:?}", entity);
        println!("triangle {:?}", triangle);

        // Get list of active adjacent chunks
        let siblings = iter_adjacent(entity, &world, relationships);
        println!("siblings: {:?}", siblings);

        let mut to_change = vec![];
        for vertex in triangle.0.iter().copied() {
            // Find the first sibling who I either share a vertex with, or my vertex is on their
            // edge.
            println!("----------------------------------------------------");

            for sibling in siblings.iter().copied() {
                let (sibling_triangle, _, _, _, _) = chunks.get(sibling)?;
                let sibling_acc_triangle = acc_triangles.get(sibling)?.as_triangle();

                let cmp_tri = sibling_triangle.cmp_bary(vertex, level.0 as u32);
                println!(
                    "point: {:.2?}, tri (size {:.3}): {:?} acc  {:?}",
                    vertex,
                    sibling_triangle.edge_arc_radius(),
                    sibling_triangle,
                    sibling_acc_triangle
                );
                println!("cmp tri: {cmp_tri:.2?}");
                match cmp_tri {
                    TriangleCmp::Corner(i) => {
                        to_change.push(sibling_acc_triangle.0[i]);
                        break;
                    }
                    TriangleCmp::Edge { v0, v1, t } => {
                        let v = sibling_acc_triangle.0[v0].lerp(sibling_acc_triangle.0[v1], t);

                        to_change.push(v);
                        break;
                    }
                    TriangleCmp::Outside | TriangleCmp::Inside => {}
                }
            }
        }
        if to_change.len() < 3 {
            println!(">>> CHUNKS <<<");
            let mut s = VecDeque::from(world.root_chunks);
            while let Some(e) = s.pop_front() {
                let (triangle, _, children, _, d) = chunks.get(e).unwrap();
                if let Some(children) = children {
                    s.extend(children.0.iter().copied());
                }
                println!("{e:?} {d:?} {triangle:?}");
            }
        }

        assert_eq!(
            to_change.len(),
            3,
            "All vertices should have a matching triangle, since we've collected all adjacent triangles above. {:?}",
            triangle,
        );

        // Apply vertex changes
        let mesh = mesh_handles.get(entity).expect("mesh");
        let mut mesh = meshes
            .get_mut(mesh.id())
            .expect("Have a handle, so mesh should exist");

        let positions = mesh
            .attribute_mut(Mesh::ATTRIBUTE_POSITION)
            .expect("Mesh should always have positions");
        let VertexAttributeValues::Float32x3(positions) = positions else {
            panic!("Unexpected data type");
        };

        let mut acc_triangle = acc_triangles.get_mut(entity)?;
        for (i, new_value) in to_change.iter().enumerate() {
            // Update mesh
            positions[i] = new_value.to_array();
            // Update acc triangle
            acc_triangle.0[i] = *new_value;
        }
    }

    Ok(())
}

fn subdivide_random_chunks(
    mut commands: Commands,
    chunks: Query<(Entity, &Triangle), Without<ChildrenChunks>>,
) {
    const MAX_CHUNKS: usize = 1;
    const MIN_RADIUS: f32 = 0.05;

    let mut rng = rand::rng();
    let indices = chunks
        .iter()
        .filter(|(_, t)| t.corner_arc_radius() >= MIN_RADIUS)
        .sample(&mut rng, MAX_CHUNKS);
    indices.into_iter().for_each(|(e, _)| {
        commands.trigger(SubdivideChunk(e));
    });
}

fn subdivide_smallest_chunks(
    mut commands: Commands,
    chunks: Query<(Entity, &Triangle), Without<ChildrenChunks>>,
) {
    const MAX_CHUNKS: usize = 1;
    const MIN_RADIUS: f32 = 0.05;

    chunks
        .iter()
        .filter(|(_, t)| t.corner_arc_radius() >= MIN_RADIUS)
        .sorted_by(|(_, t0), (_, t1)| t0.corner_arc_radius().total_cmp(&t1.corner_arc_radius()))
        .take(MAX_CHUNKS)
        .for_each(|(e, _)| {
            commands.trigger(SubdivideChunk(e));
        });
}

fn draw_entities(mut gizmos: Gizmos, chunks: Query<(Entity, &Triangle)>) {
    for (entity, triangle) in chunks {
        let translation = triangle.centre().to_vec3();

        gizmos.text(
            Isometry3d::new(translation * 1.1, Quat::IDENTITY),
            &format!("{entity:?}"),
            0.2 * triangle.corner_arc_radius(),
            Vec2::ZERO,
            Color::WHITE,
        );
    }
}

pub struct ChunkPlugin;

impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_world)
            .add_observer(subdivide_chunk)
            .add_systems(
                Update,
                (
                    // subdivide_random_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    subdivide_smallest_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    adjust_mesh_height.run_if(input_just_pressed(KeyCode::KeyL)),
                ),
            )
            .add_systems(Update, draw_entities);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        assert_matches,
        f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI},
    };

    use glam::Vec3A;

    use crate::chunks::{EPS, Triangle, TriangleCmp, arc_distance};

    fn almost_equal(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_arc_dist() {
        // Unit length
        let a = Vec3A::Y;
        let b = Vec3A::X;
        assert!(almost_equal(arc_distance(a, b), FRAC_PI_2));
        let a = Vec3A::X;
        assert!(almost_equal(arc_distance(a, b), 0.));
        let a = Vec3A::NEG_X;
        assert!(almost_equal(arc_distance(a, b), PI));

        // Different length
        let a = Vec3A::Y;
        let b = Vec3A::X * 2.45;
        assert!(almost_equal(arc_distance(a, b), FRAC_PI_2));
        let a = Vec3A::X;
        assert!(almost_equal(arc_distance(a, b), 0.));
        let a = Vec3A::NEG_X;
        assert!(almost_equal(arc_distance(a, b), PI));

        // Off-axis
        let a = Vec3A::new(1., 1., 1.).normalize();
        let b = Vec3A::new(1., 1., -1.);
        assert!(
            almost_equal(arc_distance(a, b), FRAC_1_SQRT_2.atan() * 2.),
            "{} {}",
            arc_distance(a, b),
            FRAC_1_SQRT_2.atan()
        );
    }

    #[test]
    fn test_cmp_edge_cases() {
        // case failed due to precision loss during projection to triangle axes
        let triangle0 = Triangle([
            Vec3A::new(0.32211334, 0.39611205, 0.85984784), // shared
            Vec3A::new(0.38071868, 0.41345215, 0.82710975),
            Vec3A::new(0.3353182, 0.46525362, 0.81920743), // shared
        ]);
        let triangle1 = Triangle([
            Vec3A::new(0.2763932, 0.4472136, 0.8506508),
            Vec3A::new(0.32211334, 0.39611205, 0.85984784), // shared
            Vec3A::new(0.3353182, 0.46525362, 0.81920743),  // shared
        ]);
        let t0_t1 = triangle0.0.map(|v| triangle1.cmp_bary(v, 4));
        let t1_t0 = triangle1.0.map(|v| triangle0.cmp_bary(v, 4));
        assert_matches!(
            t0_t1,
            [
                TriangleCmp::Corner(_),
                TriangleCmp::Outside,
                TriangleCmp::Corner(_),
            ]
        );
        assert_matches!(
            t1_t0,
            [
                TriangleCmp::Outside,
                TriangleCmp::Corner(_),
                TriangleCmp::Corner(_),
            ]
        );

        // Case failes as triangles share a vertex after being reflexed through origin
        let triangle0 = Triangle([
            Vec3A::new(0.28, 0.45, 0.85),
            Vec3A::new(0.59, 0.00, 0.81),
            Vec3A::new(0.69, 0.53, 0.50),
        ]);
        let triangle1 = Triangle([
            Vec3A::new(-0.72, 0.45, -0.53),
            Vec3A::new(-0.28, -0.45, -0.85),
            Vec3A::new(-0.89, -0.45, 0.00),
        ]);
        let t0_t1 = triangle0.0.map(|v| triangle1.cmp_bary(v, 4));
        let t1_t0 = triangle1.0.map(|v| triangle0.cmp_bary(v, 4));
        assert_matches!(
            t0_t1,
            [
                TriangleCmp::Outside,
                TriangleCmp::Outside,
                TriangleCmp::Outside,
            ]
        );
        assert_matches!(
            t1_t0,
            [
                TriangleCmp::Outside,
                TriangleCmp::Outside,
                TriangleCmp::Outside,
            ]
        );
        println!("=======================================");

        // Fails as point is NaN when transformed to bary coords
        let triangle0 = Triangle([
            Vec3A::new(0.7236068, -0.4472136, 0.5257311),
            Vec3A::new(0.9510565, 0.0, 0.30901697), // failing
            Vec3A::new(0.58778524, 0.0, 0.809017),
        ]);
        let triangle1 = Triangle([
            Vec3A::new(0.2763932, 0.4472136, -0.8506508),
            Vec3A::new(0.7236068, -0.4472136, -0.5257311),
            Vec3A::new(-0.2763932, -0.4472136, -0.8506508),
        ]);

        let t0_t1 = triangle0.0.map(|v| triangle1.cmp_bary(v, 4));
        let t1_t0 = triangle1.0.map(|v| triangle0.cmp_bary(v, 4));
        assert_matches!(
            t0_t1,
            [
                TriangleCmp::Outside,
                TriangleCmp::Outside,
                TriangleCmp::Outside,
            ]
        );
        assert_matches!(
            t1_t0,
            [
                TriangleCmp::Outside,
                TriangleCmp::Outside,
                TriangleCmp::Outside,
            ]
        );
    }
}
