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
    ui_widgets::{
        Activate, SetSliderValue, SliderPrecision, SliderStep, SliderValue, ValueChange,
        slider_self_update,
    },
};
use hexasphere::shapes::IcoSphere;
use sphere_world::{
    assets::{AssetHandles, load_assets},
    camera::CameraPlugin,
    chunks::{BaseMesh, TerrainColorScale},
    mesh::apply_noise_to_mesh,
    noise::{NoiseChanged, NoiseConfig, NoiseOctave},
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
    let sphere = IcoSphere::new(30, |_| ());
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

#[derive(Component, Default, Clone)]
struct ScaleSection;
#[derive(Component, Default, Clone)]
struct NumOctavesSlider;

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
            // Scales
            (
                Node {
                    align_items: AlignItems::Stretch,
                    justify_content: JustifyContent::Center,
                    flex_direction: FlexDirection::Column,
                }
                ScaleSection
                Children [
                    Text("Global input scale"),
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
                        ScaleType::Input
                    )
                    Text("Global output scale"),
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
                        ScaleType::Out
                    )
                    Text("# Octaves"),
                    (
                        @FeathersSlider {
                            @min: 1.,
                            @max: 8.,
                        }
                        SliderStep(1.)
                        SliderPrecision(0)
                        NumOctavesSlider
                        on(move |change: On<ValueChange<f32>>, mut config: ResMut<NoiseConfig>|{
                            let value = change.value as usize;
                            if config.octaves.len() != value {
                                config.resize_octaves(value);
                            }
                        })
                    )
                ]
            ),
        ]
    };

    commands.spawn_scene(left_panel);
}

#[derive(Component, Clone, Default)]
struct OctaveIndex(usize);

#[derive(Component, Clone, Default, FromTemplate)]
enum ScaleType {
    #[default]
    Input,
    Out,
}

#[derive(Component, Clone, Default)]
struct OctaveSection;

fn octave_slider(index: usize, input_val: f32, output_val: f32) -> impl Scene {
    bsn! {
        Node {
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Center,
            flex_direction: FlexDirection::Column,
        }
        OctaveIndex(index)
        OctaveSection
        Children [
            Text({ format!("=== Octave {index} ===") }),
            Text("Input scale"),
            (
                @FeathersSlider {
                    @min: {-5_f32},
                    @max: 5.,
                }
                SliderStep(0.1)
                SliderPrecision(1)
                SliderValue(input_val)
                on(move |change: On<ValueChange<f32>>, mut config: ResMut<NoiseConfig>|{
                    config.octaves[index].input_scale = 2_f64.powf(change.value as f64);
                })
                OctaveIndex(index)
                ScaleType::Input
            )
            Text("Output scale"),
            (
                @FeathersSlider {
                    @min: {-5_f32},
                    @max: 5.,
                }
                SliderValue(output_val)
                SliderStep(0.1)
                SliderPrecision(1)
                on(move |change: On<ValueChange<f32>>, mut config: ResMut<NoiseConfig>|{
                    config.octaves[index].output_scale = 2_f64.powf(change.value as f64);
                })
                OctaveIndex(index)
                ScaleType::Out
            )
        ]
    }
}

/// Update slider values when config changes
fn update_gui(
    config: Res<NoiseConfig>,
    scale_sliders: Query<(Entity, Option<&OctaveIndex>, &ScaleType)>,
    octave_sections: Query<(Entity, &OctaveIndex), With<OctaveSection>>,
    scale_section: Single<Entity, With<ScaleSection>>,
    num_octaves_slider: Single<Entity, With<NumOctavesSlider>>,
    mut commands: Commands,
) {
    // Update number of octave sections
    for (entity, index) in octave_sections {
        if index.0 >= config.octaves.len() {
            // Despawn section if # octaves has shrunk
            info!("despawning octave {} {}", index.0, entity);
            commands.entity(entity).despawn();
            continue;
        }
    }

    // Spawn some new sections if # octaves has increased
    for (index, octave) in config
        .octaves
        .iter()
        .enumerate()
        .skip(octave_sections.count())
    {
        info!("spawning octave {}", index);
        let slider = octave_slider(index, octave.input_scale as f32, octave.output_scale as f32);
        commands.spawn_scene(slider).insert(ChildOf(*scale_section));
    }

    // Update slider
    commands
        .entity(*num_octaves_slider)
        .insert(SliderValue(config.octaves.len() as f32));

    // Scale sliders
    for (entity, index, value) in scale_sliders {
        let new_value = if let Some(index) = index {
            if index.0 >= config.octaves.len() {
                // This slider has been removed above, but changes not yet propagated to world
                continue;
            }

            match value {
                ScaleType::Input => config.octaves[index.0].input_scale,
                ScaleType::Out => config.octaves[index.0].output_scale,
            }
        } else {
            match value {
                ScaleType::Input => config.input_scale,
                ScaleType::Out => config.output_scale,
            }
        };

        let new_value = new_value.log2();

        // Insert new component rather than using `SetSliderValue` event as we don't want to
        // re-trigger things that happen when the user moves the slider
        commands
            .entity(entity)
            .insert(SliderValue(new_value as f32));
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
        .add_systems(Update, update_gui.run_if(resource_changed::<NoiseConfig>))
        // Others
        .add_plugins(SunPlugin)
        .add_systems(PreStartup, load_assets)
        .add_plugins(CameraPlugin)
        .add_systems(Startup, (init_world,).chain())
        .run();
}
