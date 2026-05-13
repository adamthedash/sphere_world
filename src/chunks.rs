use std::{
    collections::VecDeque,
    f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, PI},
};

use bevy::{
    input::common_conditions::input_just_pressed, mesh::VertexAttributeValues,
    platform::collections::HashSet, prelude::*,
};
use glam::{Vec2, Vec3A};
use hexasphere::shapes::IcoSphere;
use itertools::Itertools;
use rand::seq::IteratorRandom;

use crate::{
    assets::AssetHandles,
    math::arc_distance,
    triangle::{Triangle, TrianglePointCmp, TriangleTriangleCmp},
};

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

#[derive(Component, Debug)]
pub struct AccTriangle(pub [Vec3A; 3]);

impl AccTriangle {
    /// Conversion for all of the methods
    pub fn as_triangle(&self) -> Triangle {
        Triangle::new(self.0)
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

/// The entity that has this component is adjacent to the entities within. This entity is at an
/// equal or higher LOD than the others.
#[derive(Component)]
pub struct AdjacentUp(Vec<Entity>);

/// The entity that has this component is adjacent to the entities within. This entity is at an
/// equal or lower LOD than the others.
#[derive(Component)]
pub struct AdjacentDown(Vec<Entity>);

#[derive(EntityEvent)]
pub struct CalcAdjacent {
    entity: Entity,
    is_recalc: bool,
}

fn calc_adjacent_chunks(
    event: On<CalcAdjacent>,
    mut commands: Commands,
    world: Res<WorldRoot>,
    relationships: Query<(
        &Triangle,
        &SubdivisionLevel,
        Option<&ChildrenChunks>,
        Option<&ParentChunk>,
    )>,
    mut adj_downs: Query<&mut AdjacentDown>,
) {
    let span = info_span!("calc_adjacent_chunks").entered();
    let mut queue = VecDeque::from(world.root_chunks);

    let (test_triangle, test_level, _, parent) = relationships.get(event.entity).unwrap();

    let mut adjacent_chunks = vec![];
    while let Some(candidate) = queue.pop_front() {
        let (triangle, level, children, _) = relationships.get(candidate).unwrap();

        let cmp = test_triangle.cmp_triangle(triangle, test_level.0.max(level.0) as u32);

        match cmp {
            TriangleTriangleCmp::Ancestor => {
                debug!("{:?} --[ancestor of]-> {:?}", candidate, event.entity);

                let children = children.expect("Ancestor chunks must have children");
                queue.extend(children.0.iter().copied());
            }
            TriangleTriangleCmp::Unc => {
                adjacent_chunks.push(candidate);
                if let Some(children) = children {
                    queue.extend(children.0.iter().copied());
                }
            }
            TriangleTriangleCmp::Same
            | TriangleTriangleCmp::Unrelated
            | TriangleTriangleCmp::Sibling => {}
        }
    }

    debug!("adjacent: {:?} -> {:?}", event.entity, adjacent_chunks);

    // Add to self
    commands
        .entity(event.entity)
        .insert(AdjacentUp(adjacent_chunks.clone()));

    // Add to adjacent
    for e in adjacent_chunks {
        if let Ok(mut adj_down) = adj_downs.get_mut(e) {
            // Add to existing list
            if !adj_down.0.contains(&event.entity) {
                adj_down.0.push(event.entity);
            }
        } else {
            // Add new component
            commands.entity(e).insert(AdjacentDown(vec![event.entity]));
        }
    }

    if !event.is_recalc {
        // NOTE: Edge case:
        //  1) N exists on left, N+1 exists on right
        //  2) N+2 on right is created
        //  3) N+1 on left is created
        //  -> N+2 on right needs to have N+1 inserted or all recalculated
        //  All chunks who were previously adjacent to N+1 on right need their adj recalculated
        let parent = parent.expect("This chunk came from a subdivision, so should have a parent");
        if let Ok(parent_adj_down) = adj_downs.get(parent.0) {
            for &e in &parent_adj_down.0 {
                debug!("Recalculating adjacent for {e:?}");
                commands.trigger(CalcAdjacent {
                    entity: e,
                    // Prevents infinite loops
                    is_recalc: true,
                });
            }
        }
    }

    drop(span)
}

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
            pos: ChunkPos(t.centre.normalize()),
            triangle: t,
            mesh: Mesh3d(mesh),
            transform: Transform::IDENTITY,
            material: material.clone(),
            acc_triangle: AccTriangle(t.vertices),
            subdivision: SubdivisionLevel(level.0 + 1),
        }
    });

    // Spawn chunk x4
    let children = new_bundles.map(|b| {
        commands
            // Spawn as disabled so we don't get flickering when they're hidden next frame
            .spawn((b, Visibility::Hidden))
            .id()
    });

    debug!("Subdivided chunk {:?} -> {:?}", event.0, children);

    // Add children to this chunk
    commands
        .entity(event.0)
        .add_related::<ParentChunk>(&children);

    // Trigger adjacency updates for new chunks
    children.iter().for_each(|&e| {
        commands.trigger(CalcAdjacent {
            entity: e,
            is_recalc: false,
        })
    });

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
        .map(|indices| Triangle::new(indices.map(|i| vertices[i as usize])));

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

            let pos = triangle.centre.normalize();

            commands
                .spawn((
                    ChunkBundle {
                        pos: ChunkPos(pos),
                        triangle,
                        mesh: Mesh3d(mesh),
                        transform: Transform::IDENTITY,
                        material: MeshMaterial3d(assets.hue_material.clone()),
                        acc_triangle: AccTriangle(triangle.vertices),
                        subdivision: SubdivisionLevel(0),
                    },
                    Visibility::Hidden,
                ))
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
    visibility: Query<&Visibility>,
    adj_ups: Query<&AdjacentUp>,
) -> Vec<Entity> {
    let span = info_span!("iter_adjacent").entered();

    let adjacent = adj_ups.get(entity).expect("No adj");

    let enabled = adjacent
        .0
        .iter()
        .filter(|candidate| {
            let visible = visibility.get(**candidate).unwrap();

            matches!(visible, Visibility::Visible)
        })
        .copied()
        .collect::<Vec<_>>();

    drop(span);

    enabled
}

fn adjust_mesh_height(
    world: Res<WorldRoot>,
    mesh_handles: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut acc_triangles: Query<&mut AccTriangle>,
    chunks: Query<(
        &Triangle,
        &SubdivisionLevel,
        Option<&ChildrenChunks>,
        Option<&ParentChunk>,
        &Visibility,
    )>,
    visibility: Query<&Visibility>,
    adj_ups: Query<&AdjacentUp>,
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
        let (triangle, level, children, _, visible) = chunks.get(entity)?;

        if matches!(visible, Visibility::Hidden) {
            // This one is disabled, so it doesn't need adjusting. Adjust children instead
            if let Some(children) = children {
                queue.extend(children.0.iter().copied());
            }

            continue;
        }

        // Get list of active adjacent chunks
        let siblings = iter_adjacent(entity, visibility, adj_ups);
        let span = info_span!("adjust_mesh_rest").entered();

        let mut new_acc_triangle = triangle.vertices;
        for (vertex, acc) in triangle
            .vertices
            .iter()
            .copied()
            .zip(new_acc_triangle.iter_mut())
        {
            // Find the first sibling who I either share a vertex with, or my vertex is on their
            // edge.

            for sibling in siblings.iter().copied() {
                let (sibling_triangle, _, _, _, _) = chunks.get(sibling)?;
                let sibling_acc_triangle = acc_triangles.get(sibling)?.as_triangle();

                let cmp = sibling_triangle.cmp_point(vertex, level.0 as u32);
                match cmp {
                    TrianglePointCmp::Corner(i) => {
                        *acc = sibling_acc_triangle.vertices[i];
                        break;
                    }
                    TrianglePointCmp::Edge { v0, v1, t } => {
                        *acc = sibling_acc_triangle.vertices[v0]
                            .lerp(sibling_acc_triangle.vertices[v1], t);
                        break;
                    }
                    TrianglePointCmp::Outside | TrianglePointCmp::Inside => {}
                }
            }
        }

        // Update mesh
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
        *positions.as_mut_array().unwrap() = new_acc_triangle.map(|v| v.to_array());

        // Update acc triangle
        let mut acc_triangle = acc_triangles.get_mut(entity)?;
        acc_triangle.0 = new_acc_triangle;
        drop(span);
    }

    Ok(())
}

fn subdivide_random_chunks(
    mut commands: Commands,
    chunks: Query<(Entity, &SubdivisionLevel), Without<ChildrenChunks>>,
) {
    const MAX_CHUNKS: usize = 1;

    let mut rng = rand::rng();
    chunks
        .iter()
        .filter(|(_, l)| l.0 < MAX_LOD_LEVEL)
        .sample(&mut rng, MAX_CHUNKS)
        .into_iter()
        .for_each(|(e, _)| {
            commands.trigger(SubdivideChunk(e));
        });
}

fn subdivide_smallest_chunks(
    mut commands: Commands,
    chunks: Query<(Entity, &SubdivisionLevel), Without<ChildrenChunks>>,
) {
    const MAX_CHUNKS: usize = 1;

    chunks
        .iter()
        .filter(|(_, l)| l.0 < MAX_LOD_LEVEL)
        .sorted_by_key(|(_, l)| -(l.0 as i32))
        .take(MAX_CHUNKS)
        .for_each(|(e, _)| {
            commands.trigger(SubdivideChunk(e));
        });
}

const LOD_BORDERS: [f32; 5] = [
    FRAC_PI_2, // 90+ degrees
    FRAC_PI_3, // 60+ degrees
    FRAC_PI_4, // 45+ degrees
    FRAC_PI_6, // 30+ degrees
    FRAC_PI_8, // 22.5+ degrees
               // 0+ degrees
]
.map(const |x| x / 2.);
const MAX_LOD_LEVEL: usize = LOD_BORDERS.len() + 1;

fn subdivide_close_chunks(
    mut commands: Commands,
    camera: Single<&Transform, With<Player>>,
    chunks: Query<(Entity, &Triangle, &SubdivisionLevel), Without<ChildrenChunks>>,
) {
    chunks
        .iter()
        .filter(|(_, _, l)| l.0 < MAX_LOD_LEVEL)
        .filter(|(_, t, l)| {
            // Nearest vertex
            let distance = t
                .vertices
                .iter()
                .map(|&v| arc_distance(v, camera.translation.to_vec3a()))
                .min_by(f32::total_cmp)
                .unwrap();

            let border = *LOD_BORDERS.get(l.0).unwrap_or(&0.);

            distance < border
        })
        .for_each(|(e, _, _)| {
            debug!("Subdividing chunk: {e:?}");
            commands.trigger(SubdivideChunk(e));
        });
}

/// Show higher LODs near, lower LODs far
fn toggle_lods(
    camera: Single<&Transform, With<Player>>,
    world: Res<WorldRoot>,
    mut chunks: Query<(&Triangle, &SubdivisionLevel, &mut Visibility)>,
    children: Query<&ChildrenChunks>,
) -> Result {
    let mut queue = VecDeque::from(world.root_chunks);

    while let Some(entity) = queue.pop_front() {
        let (triangle, level, mut visible) = chunks.get_mut(entity)?;

        // Nearest vertex
        let distance = triangle
            .vertices
            .iter()
            .map(|&v| arc_distance(v, camera.translation.to_vec3a()))
            .min_by(f32::total_cmp)
            .unwrap();

        let border = *LOD_BORDERS.get(level.0).unwrap_or(&0.);

        let should_show = distance >= border;

        if should_show {
            // Show self
            debug!("showing chunk {entity:?}");
            *visible = Visibility::Visible;

            // Hide all descendents
            for child in children.iter_descendants(entity) {
                let (_, _, mut visible) = chunks.get_mut(child)?;
                if matches!(*visible, Visibility::Visible) {
                    debug!("hiding child chunk {entity:?}");
                    *visible = Visibility::Hidden;
                }
            }
        } else {
            if let Ok(children) = children.get(entity) {
                // Hide self
                debug!("hiding chunk {entity:?}");
                *visible = Visibility::Hidden;

                // Queue all children for checking
                queue.extend(children.0.clone());
            } else {
                // No children, so we'll have to show this one at close range
                debug!("showing chunk {entity:?} (last layer)");
                *visible = Visibility::Visible;
            }
        }
    }

    Ok(())
}

fn draw_gizmos(mut gizmos: Gizmos, chunks: Query<(Entity, &AccTriangle)>) {
    const RED: Color = Color::srgb(1., 0., 0.);
    const GREEN: Color = Color::srgb(0., 1., 0.);
    const BLUE: Color = Color::srgb(0., 0., 1.);

    // World axes
    let axes = [
        (Vec3::X, RED, "X"),
        (Vec3::Y, BLUE, "Y"),
        (Vec3::Z, GREEN, "Z"),
    ];

    for (p, c, name) in axes {
        gizmos.line(Vec3::ZERO, p * 10., c);
        gizmos.text(
            Isometry3d::from_translation(p * 2. + p.any_orthonormal_vector() * 0.2),
            name,
            0.2,
            Vec2::ZERO,
            c,
        );
    }

    // Triangle faces
    for (entity, triangle) in chunks {
        let triangle = triangle.as_triangle();
        let translation = triangle.centre + triangle.normal * 0.01;

        // Local transform matrix
        let forward = triangle.normal.to_vec3();
        let right = Vec3::Y.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let mat = Mat3::from_cols(right, up, forward);
        let rotation = Quat::from_mat3(&mat);

        // Local axes
        let scale = triangle.edge_arc_radius();
        let t = translation.to_vec3();
        gizmos.line(t, t + forward * scale * 0.2, GREEN);
        gizmos.line(t, t + right * scale * 0.2, RED);
        gizmos.line(t, t + up * scale * 0.2, BLUE);

        gizmos.text(
            Isometry3d::new(translation.to_vec3(), rotation),
            &format!("{entity:?}"),
            0.15 * scale,
            Vec2::ZERO,
            Color::WHITE,
        );
    }
}

#[derive(Component)]
struct Player;

fn init_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetHandles>,
) {
    let mesh = Sphere::new(0.1).mesh().uv(32, 18);
    let mesh = meshes.add(mesh);
    commands.spawn((
        Player,
        Transform::from_translation(Vec3::X * 2.),
        Mesh3d(mesh),
        MeshMaterial3d(assets.hue_material.clone()),
    ));
}

fn move_player(mut player: Single<&mut Transform, With<Player>>, time: Res<Time>) {
    let rot = Quat::from_rotation_z(PI * 0.1 * time.delta_secs());

    player.rotate_around(Vec3::ZERO, rot);
}

pub struct ChunkPlugin;

impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_world)
            .add_observer(subdivide_chunk)
            .add_observer(calc_adjacent_chunks)
            // Manual systems
            .add_systems(
                Update,
                (
                    subdivide_random_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    // subdivide_smallest_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    // subdivide_close_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    adjust_mesh_height.run_if(input_just_pressed(KeyCode::KeyL)),
                ),
            )
            // Automagic LOD stuff
            .add_systems(
                Update,
                ((
                    subdivide_close_chunks, //
                    toggle_lods,
                    adjust_mesh_height,
                )
                    .chain(),),
            )
            .add_systems(Startup, init_player)
            .add_systems(Update, move_player);

        // .add_systems(Update, draw_gizmos);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        assert_matches,
        f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI},
    };

    use glam::Vec3A;

    use crate::{
        chunks::{Triangle, TrianglePointCmp, arc_distance},
        math::almost_equal,
    };

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
        let triangle0 = Triangle::new([
            Vec3A::new(0.32211334, 0.39611205, 0.85984784), // shared
            Vec3A::new(0.38071868, 0.41345215, 0.82710975),
            Vec3A::new(0.3353182, 0.46525362, 0.81920743), // shared
        ]);
        let triangle1 = Triangle::new([
            Vec3A::new(0.2763932, 0.4472136, 0.8506508),
            Vec3A::new(0.32211334, 0.39611205, 0.85984784), // shared
            Vec3A::new(0.3353182, 0.46525362, 0.81920743),  // shared
        ]);

        let t0_t1 = triangle0.vertices.map(|v| triangle1.cmp_point(v, 4));
        let t1_t0 = triangle1.vertices.map(|v| triangle0.cmp_point(v, 4));
        assert_matches!(
            t0_t1,
            [
                TrianglePointCmp::Corner(_),
                TrianglePointCmp::Outside,
                TrianglePointCmp::Corner(_),
            ]
        );
        assert_matches!(
            t1_t0,
            [
                TrianglePointCmp::Outside,
                TrianglePointCmp::Corner(_),
                TrianglePointCmp::Corner(_),
            ]
        );

        // Case failes as triangles share a vertex after being reflexed through origin
        let triangle0 = Triangle::new([
            Vec3A::new(0.28, 0.45, 0.85),
            Vec3A::new(0.59, 0.00, 0.81),
            Vec3A::new(0.69, 0.53, 0.50),
        ]);
        let triangle1 = Triangle::new([
            Vec3A::new(-0.72, 0.45, -0.53),
            Vec3A::new(-0.28, -0.45, -0.85),
            Vec3A::new(-0.89, -0.45, 0.00),
        ]);
        let t0_t1 = triangle0.vertices.map(|v| triangle1.cmp_point(v, 4));
        let t1_t0 = triangle1.vertices.map(|v| triangle0.cmp_point(v, 4));
        assert_matches!(
            t0_t1,
            [
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
            ]
        );
        assert_matches!(
            t1_t0,
            [
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
            ]
        );
        println!("=======================================");

        // Fails as point is NaN when transformed to bary coords
        let triangle0 = Triangle::new([
            Vec3A::new(0.7236068, -0.4472136, 0.5257311),
            Vec3A::new(0.9510565, 0.0, 0.30901697), // failing
            Vec3A::new(0.58778524, 0.0, 0.809017),
        ]);
        let triangle1 = Triangle::new([
            Vec3A::new(0.2763932, 0.4472136, -0.8506508),
            Vec3A::new(0.7236068, -0.4472136, -0.5257311),
            Vec3A::new(-0.2763932, -0.4472136, -0.8506508),
        ]);

        let t0_t1 = triangle0.vertices.map(|v| triangle1.cmp_point(v, 4));
        let t1_t0 = triangle1.vertices.map(|v| triangle0.cmp_point(v, 4));
        assert_matches!(
            t0_t1,
            [
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
            ]
        );
        assert_matches!(
            t1_t0,
            [
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
            ]
        );
    }
}
