use std::collections::VecDeque;

use bevy::{
    asset::RenderAssetUsages,
    ecs::entity_disabling::Disabled,
    input::common_conditions::input_just_pressed,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use glam::Vec3A;
use hexasphere::shapes::IcoSphere;
use itertools::Itertools;
use rand::seq::IteratorRandom;

use crate::uv_debug_texture;

pub struct World {
    /// Each chunk is one of the base Icosohedron faces
    /// 0-5: Top pentagon
    /// 5-10: Top ring
    /// 10-15: Bottom ring
    /// 15-20: Bottom pentagon
    pub chunks: [Chunk; 20],
}

impl World {
    /// Get the highest resolution chunk which contains this chunk
    pub fn get_chunk(&self, point: Vec3A) -> &Chunk {
        let mut root_chunk = self
            .chunks
            .iter()
            .find(|c| c.contains(point))
            .expect("There should always be a root chunk since we're a sphere");

        while let Some(chunk) = root_chunk.get_sub_chunk(point) {
            root_chunk = chunk;
        }

        root_chunk
    }

    // Get the lowest resolution chunk which shares a vertex with this one
    // fn get_chunk_for_corner(&self, vertex: Vec3A) -> &Chunk {
    //     let mut root_chunk = self
    //         .chunks
    //         .iter()
    //         .find(|c| c.contains(vertex))
    //         .expect("There should always be a root chunk since we're a sphere");
    //
    //     // Find the closest edge. If area is 0 then it's colinear
    // }
}

impl Default for World {
    fn default() -> Self {
        // Base sphere
        let sphere = IcoSphere::new(0, |_| ());

        // Create chunks
        let indices = sphere.get_all_indices();
        let vertices = sphere.raw_points();

        let chunks = indices
            .chunks_exact(3)
            .map(|c| {
                let c: [_; 3] = c.try_into().unwrap();
                let triangle = c.map(|i| vertices[i as usize]);

                let midpoint = triangle.iter().sum::<Vec3A>() / 3.;

                Chunk {
                    pos: midpoint,
                    triangle,
                    mesh: (),
                    sub_chunks: None,
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self { chunks }
    }
}

#[derive(Debug)]
pub struct Chunk {
    /// Central point (unit circle)
    pub pos: Vec3A,
    /// Corners
    pub triangle: [Vec3A; 3],
    /// Mesh: Relatively high resolution compared to the chunk triangle
    ///     Maybe 16 subdivisions?
    mesh: (),
    /// This chunk subdivided. Only generated when needed
    pub sub_chunks: Option<Box<[Chunk; 4]>>,
}

/// Area of a triangle given the 3 points
fn area2(triangle: [Vec3A; 3]) -> f32 {
    let base_mid = triangle[0].midpoint(triangle[1]);
    let height2 = (triangle[2] - base_mid).length_squared();
    let width2 = (triangle[1] - triangle[0]).length_squared();

    0.25 * height2 * width2
}

impl Chunk {
    /// Area
    fn area(&self) -> f32 {
        area2(self.triangle).sqrt()
    }

    fn edges(&self) -> [(Vec3A, Vec3A); 3] {
        self.triangle
            .iter()
            .cycle()
            .map_windows(|[v0, v1]| (**v0, **v1))
            .take(3)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version
    /// point can be anywhere in space.  
    /// Returned value represents the "as the crow flies" distance at the height of point.
    /// i.e. when point is higher, distance will be longer
    fn distance(&self, point: Vec3A) -> f32 {
        // Point above chunk at point's height
        let pos = self.pos * point.length();

        (pos.cross(point).length() / pos.dot(point)).atan()
    }

    /// Checks whether the point is contained in this chunk
    /// https://www.baeldung.com/cs/check-if-point-is-in-2d-triangle#triangles-area-approach
    fn contains(&self, point: Vec3A) -> bool {
        let areas = self
            .triangle
            .iter()
            .cycle()
            .map_windows::<_, _, 2>(|&[&v0, &v1]| area2([v0, v1, point]))
            .take(3)
            .collect::<Vec<_>>();

        areas.iter().skip(1).all(|a| *a == areas[0])
    }

    /// Finds the subchunk that contains the point, if any
    fn get_sub_chunk(&self, point: Vec3A) -> Option<&Chunk> {
        let Some(chunks) = &self.sub_chunks else {
            return None;
        };

        chunks.iter().find(|c| c.contains(point))
    }

    /// Subdivides this chunk, creating 4 children
    pub fn subdivide(&mut self) {
        assert!(
            self.sub_chunks.is_none(),
            "Subdivision can only happen once!"
        );

        // Create midpoints of edges on unit sphere
        let midpoints = self.edges().map(|(v0, v1)| v0.midpoint(v1).normalize());

        // Create outer triangles
        let outer = midpoints
            .iter()
            .cycle()
            .map_windows::<_, _, 2>(|points| *points)
            .take(3)
            .zip(self.triangle.iter().cycle().skip(1))
            // CCW order
            .map(|([v0, v1], v2)| [*v2, *v1, *v0]);

        // Inner triangle is just between the new midpoints
        let inner = std::iter::once(midpoints);

        // Create chunks
        let chunks: [_; 4] = outer
            .chain(inner)
            .map(|triangle| {
                // Centre, unit sphere
                let pos = (triangle.iter().sum::<Vec3A>() / 3.).normalize();

                Chunk {
                    pos,
                    triangle,
                    mesh: (),
                    sub_chunks: None,
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        self.sub_chunks = Some(Box::new(chunks))
    }

    // Gets a basic mesh of a single triangle for this chunk
    pub fn get_mesh(&self) -> Mesh {
        let indices = Indices::U32(vec![0, 1, 2]);
        let positions = self
            .triangle
            .iter()
            .map(|t| t.to_array())
            .collect::<Vec<_>>();

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(indices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        // .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_computed_normals()
    }
}

/// Subdivide a triangle using an existing vertex buffer
/// Returns indices for the 4 new triangles, along with new vertices which need to be added to the
/// buffer. Indices will point to new vertex indices as if they have been appended to the end.
/// i.e. index < vertices.len() - existing vertices
///      index >= vertices.len() - new vertices
fn subdivide(indices: [u32; 3], vertices: &[Vec3A]) -> ([u32; 12], [Vec3A; 3]) {
    // Create midpoints of edges on unit sphere
    let midpoints: [_; 3] = indices
        .map(|i| vertices[i as usize])
        .iter()
        .cycle()
        .map_windows::<_, _, 2>(|[v0, v1]| v0.midpoint(**v1).normalize())
        .take(3)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let inner_indices = (0..3)
        .map(|i| vertices.len() as u32 + i)
        .collect::<Vec<_>>();

    // Create outer triangles
    let outer_indices = indices
        .iter()
        .cycle()
        .map_windows::<_, _, 2>(|points| *points)
        .take(3)
        .zip(inner_indices.clone())
        .map(|([i0, i1], i2)| [*i0, *i1, i2]);

    let indices: [_; 12] = outer_indices
        .flatten()
        .chain(inner_indices)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    (indices, midpoints)
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
}

#[derive(Component)]
pub struct ChunkPos(pub Vec3A);

#[derive(Component, Debug)]
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

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(indices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        // .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_computed_normals()
    }
}

#[derive(Bundle)]
pub struct ChunkBundle {
    pub pos: ChunkPos,
    pub triangle: Triangle,
    pub mesh: Mesh3d,
    pub transform: Transform,
    pub material: MeshMaterial3d<StandardMaterial>,
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

fn init_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let debug_material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        ..default()
    });

    // Base sphere
    let sphere = IcoSphere::new(0, |_| ());
    let vertices = sphere.raw_points();

    // Create chunk triangles
    let triangles = sphere
        .get_all_indices()
        .into_iter()
        .array_chunks::<3>()
        .map(|indices| Triangle(indices.map(|i| vertices[i as usize])));

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
                    material: MeshMaterial3d(debug_material.clone()),
                })
                .id()
        })
        .collect_array()
        .expect("Should be exactly 20 faces");

    commands.insert_resource(WorldRoot {
        root_chunks: chunks,
    });
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

/// Find the lowest LOD chunk whose edge is shared with this point
fn find_edge(
    world: &WorldRoot,
    chunks: Query<(&Triangle, Option<&ChildrenChunks>, Has<Disabled>)>,
    point: Vec3A,
) -> Option<(Vec3A, Vec3A)> {
    // BFS - the first one found should(?) do.
    let mut candidate_chunks = VecDeque::from(world.root_chunks);
    while let Some(entity) = candidate_chunks.pop_front() {
        let (triangle, children, disabled) = chunks.get(entity).expect("Chunk should exist");

        // Check that our point is somewhere vaguely between the two goalposts
        if triangle.edges().into_iter().any(|(v0, v1)| {
            let edge_agreement = cossim(v0, v1);
            assert!(edge_agreement > 0.);

            let v0_agreement = cossim(v0, point);
            let v1_agreement = cossim(v1, point);

            // Outside point
            v0_agreement < edge_agreement
                // Ontop of point
                || 1. - v0_agreement < EPS
                || v1_agreement < edge_agreement
                || 1. - v1_agreement < EPS
        }) {
            continue;
        }

        if !triangle.contains(point) {
            // Point outside, discard this branch
            info!("discarding, outside triangle");
            continue;
        }

        if disabled && let Some(children) = children {
            // Check children instead
            candidate_chunks.extend(children.0.clone());
            continue;
        }

        // Check if point lies on the edge of this one
        let edge = triangle
            .edges()
            .into_iter()
            .find(|(v0, v1)| coplanar(*v0, *v1, point));

        if edge.is_some() {
            return edge;
        }
    }

    None
}

// TODO: I need a more robust way of figuring this out. Really I should be using the chunk id/LOD
// layer, traversing up the tetra-tree appropriately and reading out the vertices directly rather
// than trying to do collision checks & match vertices. Floating errors are a bitch.
fn adjust_mesh_height(
    world: Res<WorldRoot>,
    mut mesh_handles: Query<&mut Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    chunks: Query<(&Triangle, Option<&ChildrenChunks>, Has<Disabled>)>,
) -> Result {
    // Traverse in breadth-first manner
    let mut queue = VecDeque::from(world.root_chunks);
    while let Some(entity) = queue.pop_front() {
        let (triangle, children, disabled) = chunks.get(entity)?;
        info!("testing: {:?} size {}", entity, triangle.area());
        // Do special stuff, only if it's visible though
        if !disabled {
            let mut to_change = vec![];

            for (i, &corner) in triangle.0.iter().enumerate() {
                info!("testing corner: {:.2?}", corner);
                if let Some(edge) = find_edge(world.as_ref(), chunks, corner) {
                    // Project point onto edge
                    let new_corner = project(edge.0, edge.1, corner);
                    info!(
                        "projected: {:.2?} onto {:.2?} -> {:.2?}",
                        corner, edge, new_corner
                    );

                    to_change.push((i, new_corner));
                }
            }

            if !to_change.is_empty() {
                let mesh = mesh_handles.get(entity)?;
                let mesh = meshes
                    .get_mut(mesh.id())
                    .expect("Have a handle, so mesh should exist");

                let positions = mesh
                    .attribute_mut(Mesh::ATTRIBUTE_POSITION)
                    .expect("Mesh should always have positions");
                let VertexAttributeValues::Float32x3(positions) = positions else {
                    panic!("Unexpected data type");
                };

                for (i, new_value) in to_change {
                    positions[i] = new_value.to_array();
                }
            }
        }

        // Add all children to the queue
        if let Some(children) = children {
            queue.extend(children.0.iter().copied());
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

    use super::Chunk;
    use crate::chunks::coplanar;

    #[test]
    fn test_chunk_subdivision() {
        let mut chunk = Chunk {
            pos: Vec3A::ZERO,
            triangle: [
                Vec3A::new(1., 0., 1.).normalize(), // right
                Vec3A::new((TAU / 3.).cos(), (TAU / 3.).sin(), 1.).normalize(), // up left
                Vec3A::new((2. * TAU / 3.).cos(), (2. * TAU / 3.).sin(), 1.).normalize(), // down left
            ],
            mesh: (),
            sub_chunks: None,
        };

        println!("{:#.2?}", chunk);
        chunk.subdivide();
        println!("{:#.2?}", chunk);
        // panic!()
    }

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
