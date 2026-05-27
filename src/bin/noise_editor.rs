//! A lightweight editor for configuring the procedural generation
use std::path::Path;

use bevy::{
    asset::RenderAssetUsages,
    feathers::{
        FeathersPlugins,
        controls::{FeathersButton, FeathersSlider},
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens,
    },
    input::common_conditions::input_just_pressed,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
    ui_widgets::{Activate, SliderPrecision, SliderStep, ValueChange, slider_self_update},
};
use hexasphere::shapes::IcoSphere;
use sphere_world::{
    assets::{AssetHandles, load_assets},
    camera::CameraPlugin,
    chunks::{BaseMesh, TerrainColorScale},
    mesh::apply_noise_to_mesh,
    noise::{NoiseChanged, NoiseConfig},
    sun::SunPlugin,
};

fn update_heightmap(
    world: Single<(&BaseMesh, &Mesh3d), With<Planet>>,
    mut meshes: ResMut<Assets<Mesh>>,
    noise_config: Res<NoiseConfig>,
) {
    info!("Updating heightmap");
    let (base_mesh, mesh) = *world;

    let noise_gen = noise_config.generator();

    // Get mesh
    let base_mesh = meshes.get(base_mesh.0.id()).unwrap().clone();

    let new_mesh = apply_noise_to_mesh(base_mesh, &noise_gen)
        .with_duplicated_vertices()
        .with_computed_normals();

    let mut mesh = meshes.get_mut(mesh.id()).unwrap();
    *mesh = new_mesh;
}

fn update_terrain_texture(
    world: Single<&Mesh3d, With<Planet>>,
    mut meshes: ResMut<Assets<Mesh>>,
    terrain_config: Res<TerrainColorScale>,
) {
    info!("Updating terrain");
    let mut mesh = meshes.get_mut(world.0.id()).unwrap();

    let vertices = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .expect("Mesh should always have positions");
    let VertexAttributeValues::Float32x3(vertices) = vertices else {
        panic!("Unexpected data type");
    };

    let heights = vertices.iter().map(|v| Vec3A::from_array(*v).length());
    let colors = heights
        .map(|h| terrain_config.sample(h).to_f32_array())
        .collect::<Vec<_>>();

    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
}

#[derive(Component)]
pub struct Planet;

fn init_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetHandles>,
    noise_config: Res<NoiseConfig>,
) {
    // Base sphere
    let sphere = IcoSphere::new(20, |_| ());
    let vertices = sphere.raw_points();

    // Create base mesh - unit sphere
    let base_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U32(sphere.get_all_indices()))
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vertices.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
    );

    // Unit sphere with height map added
    let noise_gen = noise_config.generator();
    let mut mesh = apply_noise_to_mesh(base_mesh.clone(), &noise_gen);
    mesh.compute_normals();

    // Spawn planet
    commands.spawn((
        Transform::IDENTITY,
        BaseMesh(Mesh3d(meshes.add(base_mesh))),
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(assets.hue_material.clone()),
        Planet,
    ));
}

fn gui(mut commands: Commands) {
    let left_panel = bsn! {
        Node {
            width: percent(20),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            // Save/load
            (
                Node {
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                }
                Children [
                    (
                        @FeathersButton
                        Children [ Text("Save") ThemedText ]
                        on(|_: On<Activate>, config: Res<NoiseConfig>| {
                            config.save(Path::new("noise_config.json")).unwrap();
                            info!("Saved noise config");
                        })
                    ),
                    (
                        @FeathersButton
                        Children [ Text("Load") ThemedText ]
                        on(|_: On<Activate>, mut config: ResMut<NoiseConfig>| {
                            *config = NoiseConfig::load(Path::new("noise_config.json")).unwrap();
                            info!("Loaded noise config");
                        })
                    ),
                ]
            ),
            (
                Node {
                    align_items: AlignItems::Stretch,
                    justify_content: JustifyContent::Center,
                    flex_direction: FlexDirection::Column,
                }
                Children [
                    (
                        Text("Global input scale")
                    ),
                    (
                        @FeathersSlider {
                            @min: {-5_f32},
                            @max: 5.,
                        }
                        SliderStep(0.1)
                        SliderPrecision(1)
                        on(slider_self_update)
                        on(|change: On<ValueChange<f32>>, mut config: ResMut<NoiseConfig>|{
                            config.input_scale = 2_f64.powf(change.value as f64);
                        })
                    )
                    (
                        Text("Global output scale")
                    ),
                    (
                        @FeathersSlider {
                            @min: {-5_f32},
                            @max: 5.,
                        }
                        SliderStep(0.1)
                        SliderPrecision(1)
                        on(slider_self_update)
                        on(|change: On<ValueChange<f32>>, mut config: ResMut<NoiseConfig>|{
                            config.output_scale = 2_f64.powf(change.value as f64);
                        })
                    )
                ]
            ),
        ]
    };

    commands.spawn_scene(left_panel);
}

fn octave_slider(index: usize) -> impl Scene {
    bsn! {
        Node {
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Center,
            flex_direction: FlexDirection::Column,
        }
        Children [
            (
                Text("Input scale")
            ),
            (
                @FeathersSlider {
                    @min: {-5_f32},
                    @max: 5.,
                }
                SliderStep(0.1)
                SliderPrecision(1)
                on(slider_self_update)
                on(|change: On<ValueChange<f32>>, mut config: ResMut<NoiseConfig>|{
                    config.input_scale = 2_f64.powf(change.value as f64);
                })
            )
        ]
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        // Wireframe
        .add_plugins(WireframePlugin::default())
        .add_systems(
            Update,
            { |mut c: ResMut<WireframeConfig>| c.global ^= true }
                .run_if(input_just_pressed(KeyCode::KeyW)),
        )
        // Noise
        .insert_resource(NoiseConfig::default())
        .add_systems(
            Update,
            (
                (update_heightmap, update_terrain_texture)
                    .chain()
                    .run_if(resource_changed::<NoiseConfig>),
                update_terrain_texture.run_if(resource_changed::<TerrainColorScale>),
            ),
        )
        .insert_resource(TerrainColorScale {
            sea_level: 1.,
            mountain_start: 1.1,
            snow_start: 1.2,
        })
        .add_systems(PostStartup, |mut noise: ResMut<NoiseConfig>| {
            *noise = NoiseConfig::load(Path::new("noise_config.json")).unwrap();
        })
        // UI
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .add_systems(Startup, gui)
        // Others
        .add_plugins(SunPlugin)
        .add_systems(PreStartup, load_assets)
        .add_plugins(CameraPlugin)
        .add_systems(Startup, (init_world,).chain())
        .run();
}
