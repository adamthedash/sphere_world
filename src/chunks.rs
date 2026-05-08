use std::{collections::VecDeque, f32::consts::TAU};

use bevy::{
    asset::RenderAssetUsages,
    ecs::entity_disabling::Disabled,
    input::common_conditions::input_just_pressed,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    platform::collections::{HashMap, HashSet},
    prelude::*,
};
use glam::Vec3A;
use hexasphere::shapes::IcoSphere;
use itertools::Itertools;
use rand::seq::IteratorRandom;

use crate::assets::AssetHandles;

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

const EPS: f32 = 1e-6;

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

#[derive(Component, Debug, Clone, Copy)]
pub struct Triangle(pub [Vec3A; 3]);

impl Triangle {
    fn area(&self) -> f32 {
        area2(self.0).sqrt()
    }

    fn edges(&self) -> [(Vec3A, Vec3A); 3] {
        self.0
            .iter()
            .cycle()
            .map_windows(|[v0, v1]| (**v0, **v1))
            .take(3)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Centre point on unit circle
    fn centre(&self) -> Vec3A {
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

    /// Checks whether the point is contained in this chunk
    /// https://www.baeldung.com/cs/check-if-point-is-in-2d-triangle#orientation-approach
    fn contains(&self, point: Vec3A) -> bool {
        // Triangles are defined in CW order
        // Normals point inward
        let normals = self.edges().map(|(v0, v1)| v0.cross(v1));
        let alignments = normals.map(|n| n.dot(point));
        info!("alignments: {:.2?}", alignments);

        // If we're infront of all planes, then we're inside
        alignments.iter().all(|s| *s > -EPS)
    }

    fn subdivide(&self) -> [Self; 4] {
        // Create midpoints of edges on unit sphere
        let midpoints = self.edges().map(|(v0, v1)| v0.midpoint(v1).normalize());

        // Create outer triangles
        let outer = midpoints
            .iter()
            .cycle()
            .map_windows::<_, _, 2>(|points| *points)
            .take(3)
            .zip(self.0.iter().cycle().skip(1))
            // CCW order
            .map(|([v0, v1], v2)| [*v2, *v1, *v0]);

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

#[derive(Bundle)]
pub struct ChunkBundle {
    pub pos: ChunkPos,
    pub triangle: Triangle,
    pub mesh: Mesh3d,
    pub transform: Transform,
    pub material: MeshMaterial3d<StandardMaterial>,
    pub acc_triangle: AccTriangle,
}

#[derive(Component)]
#[relationship(relationship_target = ChildrenChunks)]
pub struct ChildChunk(Entity);

#[derive(Component)]
#[relationship_target(relationship = ChildChunk)]
pub struct ChildrenChunks(Vec<Entity>);

#[derive(EntityEvent)]
pub struct SubdivideChunk(Entity);

pub fn subdivide_chunk(
    event: On<SubdivideChunk>,
    mut commands: Commands,
    chunks: Query<(&Triangle, &MeshMaterial3d<StandardMaterial>)>,
    mut meshes: ResMut<Assets<Mesh>>,
) -> Result {
    let (triangle, material) = chunks.get(event.0)?;

    let new_bundles = triangle.subdivide().map(|t| {
        let mesh = meshes.add(t.get_mesh());
        ChunkBundle {
            pos: ChunkPos(t.centre()),
            triangle: t,
            mesh: Mesh3d(mesh),
            transform: Transform::IDENTITY,
            material: material.clone(),
            acc_triangle: AccTriangle(t.0),
        }
    });

    // Spawn chunk x4
    let children = new_bundles.map(|b| commands.spawn(b).id());

    // Add children to this chunk
    commands
        .entity(event.0)
        .add_related::<ChildChunk>(&children)
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
    mut commands: Commands,
    world: Res<WorldRoot>,
    mesh_handles: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut acc_triangles: Query<&mut AccTriangle>,
    chunks: Query<(
        &Triangle,
        Option<&ChildrenChunks>,
        Option<&ChildChunk>,
        Has<Disabled>,
    )>,
) -> Result {
    let mut queue = VecDeque::new();
    // Add all base+1 chunks to the queue, base chunks don't need processing
    for entity in world.root_chunks {
        let (_, children, _, _) = chunks.get(entity)?;
        if let Some(children) = children {
            queue.extend(children.0.clone());
        }
    }

    // Traverse in breadth-first manner
    while let Some(entity) = queue.pop_front() {
        let (triangle, children, parent, disabled) = chunks.get(entity)?;
        if disabled && let Some(children) = children {
            // This one is disabled, so it doesn't need adjusting. Adjust children instead
            queue.extend(children.0.iter().copied());
            continue;
        }

        info!("testing: {:?} size {}", entity, triangle.area());

        // Get parent
        let parent = parent
            .expect("All chunks except base should have a parent")
            .0;
        let (_, _, grandparent, _) = chunks.get(parent).expect("parent");

        // Get parent's siblings
        let siblings = if let Some(grandparent) = grandparent {
            // Standard chunk
            let (_, siblings, _, _) = chunks.get(grandparent.0).expect("grandparent");

            siblings
                .expect("Just traversed up to this")
                .0
                .iter()
                .copied()
                // Only want parent' siblings, not parent
                .filter(|e| *e != parent)
                .collect::<Vec<_>>()
        } else {
            // Base chunk, need to find siblings using WorldRoot
            world.get_siblings(parent).expect("Bad base chunk").to_vec()
        };

        let mut to_change = vec![];
        for (i, vertex) in triangle.0.iter().copied().enumerate() {
            for sibling in siblings.iter().copied() {
                let sibling_acc_triangle = acc_triangles.get(entity)?.as_triangle();
                let (sibling_triangle, _, _, sibling_disabled) = chunks.get(sibling)?;

                if sibling_disabled {
                    // Since this is disabled (not visibble), we don't need to adjust for it
                    continue;
                }

                // Find vertex which shares an edge.
                let edge_radius = sibling_triangle.edge_arc_radius();

                let vertex_distance = arc_distance(sibling_triangle.centre(), vertex);
                if vertex_distance < edge_radius + EPS {
                    // This vertex is on the edge of the sibling triangle
                    // Find the midpoint of the touching edge
                    // let mut indices = [0, 1, 2];
                    // indices.sort_unstable_by(|i0, i1| {
                    //     let d0 = arc_distance(vertex, sibling_triangle.0[*i0]);
                    //     let d1 = arc_distance(vertex, sibling_triangle.0[*i1]);
                    //     d0.total_cmp(&d1)
                    // });
                    //
                    // let midpoint = sibling_acc_triangle.0[indices[0]]
                    //     .midpoint(sibling_acc_triangle.0[indices[1]]);

                    let mut sibling_vertices = sibling_triangle.0;
                    sibling_vertices.sort_unstable_by(|v0, v1| {
                        arc_distance(vertex, *v0).total_cmp(&arc_distance(vertex, *v1))
                    });
                    let midpoint = sibling_vertices[0].midpoint(sibling_vertices[1]);

                    info!("Moving vertex {vertex:?} to {midpoint:?}");
                    to_change.push((i, midpoint));
                }
            }

            if to_change.len() == i {
                // No change needed, so we reset to the base height
                to_change.push((i, vertex));
            }
        }

        // Apply vertex changes
        if !to_change.is_empty() {
            let mesh = mesh_handles.get(entity).expect("mesh");
            let mesh = meshes
                .get_mut(mesh.id())
                .expect("Have a handle, so mesh should exist");

            let positions = mesh
                .attribute_mut(Mesh::ATTRIBUTE_POSITION)
                .expect("Mesh should always have positions");
            let VertexAttributeValues::Float32x3(positions) = positions else {
                panic!("Unexpected data type");
            };

            let mut acc_triangle = acc_triangles.get_mut(entity)?;
            for (i, new_value) in to_change {
                // Update mesh
                positions[i] = new_value.to_array();
                // Update acc triangle
                acc_triangle.0[i] = new_value;
            }
        }
    }

    Ok(())
}

fn subdivide_random_chunks(
    mut commands: Commands,
    chunks: Query<Entity, (With<Triangle>, Without<ChildrenChunks>)>,
) {
    const MAX_CHUNKS: usize = 1;

    let mut rng = rand::rng();
    let indices = chunks.iter().sample(&mut rng, MAX_CHUNKS);
    indices.into_iter().for_each(|e| {
        commands.trigger(SubdivideChunk(e));
    });
}

pub struct ChunkPlugin;

impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_world)
            .add_observer(subdivide_chunk)
            .add_systems(
                Update,
                (
                    subdivide_random_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    adjust_mesh_height.run_if(input_just_pressed(KeyCode::KeyL)),
                ),
            );
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3A;

    use crate::chunks::coplanar;

    #[test]
    fn test_coplanar() {
        let a = Vec3A::new(-0.59, 0., 0.81);
        let b = Vec3A::new(-0.72, 0.45, -0.53);
        let c = Vec3A::new(-0.72, 0.45, 0.53);

        assert!(!coplanar(a, b, c), "{:?}, {:?}, {:?}", a, b, c);

        let a = Vec3A::new(-0.72, 0.45, -0.53);
        let b = Vec3A::new(-0.72, 0.45, 0.53);
        let c = Vec3A::new(-0.61, 0.37, 0.70);
        println!(
            "{:?}",
            a.normalize().cross(b.normalize()).dot(c.normalize())
        );

        assert!(coplanar(a, b, c), "{:?}, {:?}, {:?}", a, b, c);
    }
}
