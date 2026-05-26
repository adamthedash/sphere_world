use std::{
    collections::VecDeque,
    f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, PI},
    hash::Hash,
    path::Path,
};

use bevy::{
    input::common_conditions::input_just_pressed,
    mesh::VertexAttributeValues,
    platform::collections::{HashMap, HashSet},
    prelude::*,
};
use glam::{Vec2, Vec3A};
use hexasphere::shapes::IcoSphere;
use itertools::Itertools;
use num::{Integer, ToPrimitive, rational::Ratio};
use rand::{SeedableRng, seq::IteratorRandom};

use crate::{
    assets::AssetHandles,
    coordinates::bary_iterative::BaryIterative,
    math::arc_distance,
    mesh::{MESH_STEPS, MESH_SUBDIVISIONS, create_mesh},
    noise::{NoiseChanged, NoiseConfig},
    player::Player,
    triangle::{Triangle, TriangleTriangleCmp},
};

const DEBUG_ENTITY: u32 = 380;

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

/// Level of subdivision this triangle is part of. Root == 0
#[derive(Component, Debug)]
pub struct SubdivisionLevel(usize);

/// "Base" mesh before any stitching
#[derive(Component, Debug)]
pub struct BaseMesh(Mesh3d);

#[derive(Component, Debug)]
pub struct MeshOverrides(HashMap<u32, [f32; 3]>);

#[derive(Bundle)]
pub struct ChunkBundle {
    pub pos: ChunkPos,
    pub triangle: Triangle,
    pub transform: Transform,
    pub material: MeshMaterial3d<StandardMaterial>,
    pub subdivision: SubdivisionLevel,
    pub base_mesh: BaseMesh,
    pub mesh_overrides: MeshOverrides,
    pub render_mesh: Mesh3d,
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

        if event.entity.index_u32() == DEBUG_ENTITY {
            info!("{:?} cmp {:?}: {:?}", event.entity, candidate, cmp);
        }

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
    noise_config: Res<NoiseConfig>,
) -> Result {
    let (triangle, material, level) = chunks.get(event.0)?;
    let new_triangles = triangle.subdivide();

    let noise_gen = noise_config.generator();

    let new_bundles = new_triangles.map(|t| {
        let is_ccw = t.vertices[0].cross(t.vertices[1]).dot(t.vertices[2]) > 0.;
        assert!(is_ccw);

        // Base mesh is the "true" mesh for this chunk. It's created once on chunk creation
        let base_mesh = create_mesh(t.vertices, &noise_gen);

        // Acc mesh is the mesh that's rendered. It's a modified version of the base mesh
        // depending on what's happening with ajacent chunks
        let acc_mesh = base_mesh.clone();

        let base_mesh = meshes.add(base_mesh);
        let acc_mesh = meshes.add(acc_mesh);

        ChunkBundle {
            pos: ChunkPos(t.centre.normalize()),
            triangle: t,
            transform: Transform::IDENTITY,
            material: material.clone(),
            subdivision: SubdivisionLevel(level.0 + 1),
            base_mesh: BaseMesh(Mesh3d(base_mesh)),
            mesh_overrides: MeshOverrides(HashMap::new()),
            render_mesh: Mesh3d(acc_mesh),
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

fn init_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetHandles>,
    noise_config: Res<NoiseConfig>,
) {
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

    let noise_gen = noise_config.generator();

    // Spawn shape
    let chunks = triangles
        .map(|triangle| {
            let is_ccw = triangle.vertices[0]
                .cross(triangle.vertices[1])
                .dot(triangle.vertices[2])
                > 0.;
            assert!(is_ccw);
            // Base mesh is the "true" mesh for this chunk. It's created once on chunk creation
            let base_mesh = create_mesh(triangle.vertices, &noise_gen);

            // Acc mesh is the mesh that's rendered. It's a modified version of the base mesh
            // depending on what's happening with ajacent chunks
            let acc_mesh = base_mesh.clone();

            let base_mesh = meshes.add(base_mesh);
            let acc_mesh = meshes.add(acc_mesh);

            let pos = triangle.centre.normalize();

            commands
                .spawn((
                    ChunkBundle {
                        pos: ChunkPos(pos),
                        triangle,
                        transform: Transform::IDENTITY,
                        material: MeshMaterial3d(assets.hue_material.clone()),
                        subdivision: SubdivisionLevel(0),
                        base_mesh: BaseMesh(Mesh3d(base_mesh)),
                        mesh_overrides: MeshOverrides(HashMap::new()),
                        render_mesh: Mesh3d(acc_mesh),
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

    let Ok(adjacent) = adj_ups.get(entity) else {
        // Root nodes won't have this
        return vec![];
    };

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

/// Adjust height of vertices along mesh edges so they join seamlessly with their neighbours
fn sitch_meshes(
    world: Res<WorldRoot>,
    mut base_mesh_overrides: Query<&mut MeshOverrides>,
    base_mesh_handles: Query<&BaseMesh>,
    meshes: Res<Assets<Mesh>>,
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
    for entity in world.root_chunks {
        // Root chunks don't need to be processed
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

        if entity.index_u32() == DEBUG_ENTITY {
            info!("siblings {entity} -> {siblings:?}");
        }

        // Iterate over edges of this mesh and adjust it down to where the adjacent mesh is
        let mut mesh_overrides = vec![];
        for bary_self in BaryIterative::perimeter(MESH_SUBDIVISIONS) {
            let point = bary_self.to_cartesian(triangle.vertices);
            let self_index = bary_self.to_mesh_index();

            for sibling in siblings.clone() {
                let (sibling_triangle, sibling_level, _, _, _) = chunks.get(sibling)?;
                let subdivision_ratio = (level.0 - sibling_level.0) as u32;

                // Find out where this point lies in the sibling's coordinate system
                let Some(sibling_bary) = BaryIterative::from_cartesian(
                    sibling_triangle.vertices,
                    point,
                    MESH_SUBDIVISIONS + subdivision_ratio,
                ) else {
                    // Point doesn't lie on this sibling
                    continue;
                };

                // Get sibling mesh - use non-base mesh as we want to include adjustments for
                // siblings processed in previous loops of this function
                let sibling_mesh_vertices = {
                    let mesh = base_mesh_handles.get(sibling)?;
                    let positions = meshes
                        .get(mesh.0.id())
                        .expect("Have a handle, so mesh should exist")
                        .attribute(Mesh::ATTRIBUTE_POSITION)
                        .expect("Mesh should always have positions");
                    let VertexAttributeValues::Float32x3(positions) = positions else {
                        panic!("Unexpected data type");
                    };
                    positions
                };

                let sibling_mesh_overrides = &base_mesh_overrides.get(sibling)?.0;
                let get_sibling_vertex = |i: u32| {
                    // Prioritize an override if there is one
                    if let Some(v) = sibling_mesh_overrides.get(&i) {
                        return *v;
                    }

                    sibling_mesh_vertices[i as usize]
                };

                let nonzero_axes = sibling_bary
                    .weights
                    .to_array()
                    .iter()
                    .positions(|w| *w > 0)
                    .collect::<Vec<_>>();

                match *nonzero_axes.as_slice() {
                    [_i] => {
                        // Sibling corner, downsample and take point
                        let sibling_bary = sibling_bary
                            .downsample(subdivision_ratio)
                            .expect("Corners should downsample cleanly");
                        let sibling_index = sibling_bary.to_mesh_index();
                        mesh_overrides.push((self_index, get_sibling_vertex(sibling_index)));
                    }
                    [i0, i1] => {
                        // Somewhere along an edge

                        let (d0, r0) = sibling_bary.weights[i0].div_rem(&(1 << subdivision_ratio));
                        let d1 = d0 + 1;

                        // Construct sibling coordinates for 2 nearest vertices
                        let mut weights0 = [0; 3];
                        weights0[i0] = d0;
                        weights0[i1] =
                            (sibling_bary.denominator >> subdivision_ratio) - weights0[i0];
                        let sibling_bary0 = BaryIterative::new(weights0.into(), 1.);

                        let mut weights1 = [0; 3];
                        weights1[i0] = d1;
                        weights1[i1] =
                            (sibling_bary.denominator >> subdivision_ratio) - weights1[i0];
                        let sibling_bary1 = BaryIterative::new(weights1.into(), 1.);

                        // Interpolate between em
                        let index0 = sibling_bary0.to_mesh_index();
                        let index1 = sibling_bary1.to_mesh_index();
                        let t = Ratio::new(r0, 1 << subdivision_ratio);
                        let vertex0 = Vec3A::from_array(get_sibling_vertex(index0));
                        let vertex1 = Vec3A::from_array(get_sibling_vertex(index1));
                        let vertex = vertex0.lerp(vertex1, t.to_f32().unwrap());
                        mesh_overrides.push((self_index, vertex.to_array()));
                    }
                    _ => unreachable!(
                        "Point should always land on either a corner or edge of sibling"
                    ),
                }

                // Always break as we don't want to double override vertices with higher-res mesh
                // values, always take the lowest res one
                break;
            }
        }

        // Update mesh overrides
        let mut overrides = base_mesh_overrides.get_mut(entity)?;
        overrides.0.clear();
        overrides.0.extend(mesh_overrides.into_iter());
    }

    Ok(())
}

/// Push stitching updates to the render meshes
fn prepare_render_meshes(
    query: Query<(&BaseMesh, &MeshOverrides, &Mesh3d), Changed<MeshOverrides>>,
    mut meshes: ResMut<Assets<Mesh>>,
) -> Result {
    for (base_mesh, overrides, render_mesh) in query {
        // Make copy of base mesh to start with
        let mut base_mesh = meshes
            .get(base_mesh.0.id())
            .expect("Have handle, so mesh should exist.")
            .clone();

        // Apply overrides
        let vertices = base_mesh
            .attribute_mut(Mesh::ATTRIBUTE_POSITION)
            .expect("Mesh should always have positions");
        let VertexAttributeValues::Float32x3(vertices) = vertices else {
            panic!("Unexpected data type");
        };
        for (i, v) in overrides.0.iter() {
            vertices[*i as usize] = *v;
        }

        // Duplicate vertices so we can do per-face normals
        base_mesh.duplicate_vertices();
        base_mesh.compute_flat_normals();

        // Update render mesh
        *meshes
            .get_mut(render_mesh.id())
            .expect("Have handle, so mesh should exist.") = base_mesh;
    }

    Ok(())
}

fn subdivide_random_chunks(
    mut commands: Commands,
    chunks: Query<(Entity, &SubdivisionLevel), Without<ChildrenChunks>>,
) {
    const MAX_CHUNKS: usize = 1;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
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

#[derive(Resource, PartialEq)]
pub struct DrawGizmos(bool);

fn draw_gizmos(
    mut gizmos: Gizmos,
    chunks: Query<(Entity, &Triangle, &Visibility, &BaseMesh)>,
    meshes: Res<Assets<Mesh>>,
) {
    const RED: Color = Color::srgb(1., 0., 0.);
    const GREEN: Color = Color::srgb(0., 1., 0.);
    const BLUE: Color = Color::srgb(0., 0., 1.);
    const MAGENTA: Color = Color::srgb(1., 0., 1.);

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
    for (entity, triangle, visibility, base_mesh) in chunks {
        if matches!(visibility, Visibility::Hidden) {
            continue;
        }

        let centre_translation = triangle.centre.normalize() + triangle.normal * 0.1;

        // Local transform matrix
        let forward = triangle.normal.to_vec3();
        let right = Vec3::Y.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let mat = Mat3::from_cols(right, up, forward);
        let rotation = Quat::from_mat3(&mat);

        // Local axes
        let scale = triangle.edge_arc_radius();
        let t = centre_translation.to_vec3();
        gizmos.line(t, t + forward * scale * 0.2, GREEN);
        gizmos.line(t, t + right * scale * 0.2, RED);
        gizmos.line(t, t + up * scale * 0.2, BLUE);

        // Entitity ID
        gizmos.text(
            Isometry3d::new(centre_translation.to_vec3(), rotation),
            &format!("{entity:?}"),
            0.15 * scale,
            Vec2::ZERO,
            MAGENTA,
        );

        // Mesh vertex IDs
        let mesh = meshes.get(base_mesh.0.id()).unwrap();
        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        let VertexAttributeValues::Float32x3(positions) = positions else {
            unreachable!()
        };
        for (i, pos) in positions.iter().enumerate() {
            let translation = Vec3A::from_array(*pos).slerp(centre_translation, 0.1);

            gizmos.text(
                Isometry3d::new(translation.to_vec3(), rotation),
                &format!("{i:?}"),
                0.1 * scale,
                Vec2::ZERO,
                MAGENTA,
            );
        }
    }
}

const LOD_BORDERS: [f32; 2] = [
    FRAC_PI_2, // 90+ degrees
    // FRAC_PI_3, // 60+ degrees
    FRAC_PI_4, // 45+ degrees
              // FRAC_PI_6, // 30+ degrees
              // FRAC_PI_8, // 22.5+ degrees
              // 0+ degrees
]
.map(const |x| x / 2.);

// Debug - show at all times
// const LOD_BORDERS: [f32; 5] = [PI; _];

const MAX_LOD_LEVEL: usize = LOD_BORDERS.len() + 1;

pub struct ChunkPlugin;

impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app
            // Noise
            .insert_resource(NoiseConfig::default())
            .add_message::<NoiseChanged>()
            // .insert_resource(ShouldRegenerateMesh(true))
            .add_systems(
                Startup,
                (
                    |mut noise: ResMut<NoiseConfig>| {
                        *noise = NoiseConfig::load(Path::new("noise_config.json")).unwrap();
                    },
                    init_world,
                )
                    .chain(),
            )
            // Manual systems
            .add_systems(
                Update,
                (
                    subdivide_random_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    // subdivide_smallest_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    // subdivide_close_chunks.run_if(input_just_pressed(KeyCode::Space)),
                    sitch_meshes.run_if(input_just_pressed(KeyCode::KeyL)),
                ),
            )
            // Automagic LOD stuff
            .add_systems(
                Update,
                ((
                    subdivide_close_chunks, //
                    toggle_lods,
                    sitch_meshes,
                    prepare_render_meshes,
                )
                    .chain(),),
            )
            .add_observer(subdivide_chunk)
            .add_observer(calc_adjacent_chunks)
            // Debug stuff
            .insert_resource(DrawGizmos(false))
            .add_systems(
                Update,
                { |mut g: ResMut<DrawGizmos>| g.0 ^= true }
                    .run_if(input_just_pressed(KeyCode::KeyG)),
            )
            .add_systems(
                Update,
                draw_gizmos.run_if(resource_equals(DrawGizmos(true))),
            );
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
        chunks::arc_distance,
        math::almost_equal,
        triangle::{Triangle, TrianglePointCmp},
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
        let triangle0 = Triangle::new(
            [
                Vec3A::new(0.32211334, 0.39611205, 0.85984784), // shared
                Vec3A::new(0.38071868, 0.41345215, 0.82710975),
                Vec3A::new(0.3353182, 0.46525362, 0.81920743), // shared
            ]
            .map(|v| v.normalize()),
        );
        let triangle1 = Triangle::new(
            [
                Vec3A::new(0.2763932, 0.4472136, 0.8506508),
                Vec3A::new(0.32211334, 0.39611205, 0.85984784), // shared
                Vec3A::new(0.3353182, 0.46525362, 0.81920743),  // shared
            ]
            .map(|v| v.normalize()),
        );

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
        let triangle0 = Triangle::new(
            [
                Vec3A::new(0.28, 0.45, 0.85),
                Vec3A::new(0.59, 0.00, 0.81),
                Vec3A::new(0.69, 0.53, 0.50),
            ]
            .map(|v| v.normalize()),
        );
        let triangle1 = Triangle::new(
            [
                Vec3A::new(-0.72, 0.45, -0.53),
                Vec3A::new(-0.28, -0.45, -0.85),
                Vec3A::new(-0.89, -0.45, 0.00),
            ]
            .map(|v| v.normalize()),
        );
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
        let triangle0 = Triangle::new(
            [
                Vec3A::new(0.7236068, -0.4472136, 0.5257311),
                Vec3A::new(0.9510565, 0.0, 0.30901697), // failing
                Vec3A::new(0.58778524, 0.0, 0.809017),
            ]
            .map(|v| v.normalize()),
        );
        let triangle1 = Triangle::new(
            [
                Vec3A::new(0.2763932, 0.4472136, -0.8506508),
                Vec3A::new(0.7236068, -0.4472136, -0.5257311),
                Vec3A::new(-0.2763932, -0.4472136, -0.8506508),
            ]
            .map(|v| v.normalize()),
        );

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
