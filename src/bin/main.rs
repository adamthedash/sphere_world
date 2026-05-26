use std::path::Path;

use bevy::{
    ecs::error::Result,
    input::common_conditions::input_just_pressed,
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
};
use bevy_egui::{egui::Slider, prelude::*};
use glam::Vec3A;
use hexasphere::shapes::IcoSphere;
use sphere_world::{
    assets::load_assets,
    camera::CameraPlugin,
    chunks::ChunkPlugin,
    noise::{NoiseChanged, NoiseConfig, NoiseConfigWidget},
    player::PlayerPlugin,
    sun::SunPlugin,
};

struct SphereData {
    origin: Vec3A,
    normal: Vec3A,
    uv: [f32; 2],
    height: f32,
}

#[derive(Resource)]
struct Planet(IcoSphere<SphereData>);

#[derive(Resource, PartialEq, Eq)]
struct ShouldRegenerateMesh(bool);

fn update_mesh(
    mut meshes: ResMut<Assets<Mesh>>,
    mesh: Single<&mut Mesh3d>,
    noise_config: Res<NoiseConfig>,
    planet: Res<Planet>,
    mut should_regen: ResMut<ShouldRegenerateMesh>,
) {
    info!("Regenerating mesh");
    let mut mesh = meshes.get_mut(mesh.0.id()).expect("mesh not found");

    // Recompute height map & normals based with new noise function
    let noise = noise_config.generator();

    let points = planet
        .0
        .raw_data()
        .iter()
        .map(|d| d.origin.to_array())
        .collect::<Vec<_>>();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, points);
    mesh.compute_normals();

    should_regen.0 = false;
}

fn draw_ui(
    mut contexts: EguiContexts,
    mut noise_config: ResMut<NoiseConfig>,
    mut regen_mesh: ResMut<ShouldRegenerateMesh>,
    mut regen_mesh_writer: MessageWriter<NoiseChanged>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    egui::SidePanel::right("side_panel").show(ctx, |ui| {
        let mut num_subdivisions = 1;
        ui.add(Slider::new(&mut num_subdivisions, 1..=20).text("# Subdivisions"));

        ui.horizontal(|ui| -> Result {
            let config_path = Path::new("noise_config.json");
            if ui.button("save").clicked() {
                noise_config.save(config_path)?;
            }
            if ui.button("load").clicked() {
                *noise_config = NoiseConfig::load(config_path)?;
                regen_mesh.0 = true;
            }

            Ok(())
        });

        ui.add(NoiseConfigWidget::new(
            &mut noise_config,
            &mut regen_mesh_writer,
        ));
    });

    Ok(())
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
        // UI
        // .add_plugins(EguiPlugin::default())
        // .add_systems(EguiPrimaryContextPass, draw_ui)
        .insert_resource(ClearColor(Color::BLACK))
        // Assets
        .add_systems(PreStartup, load_assets)
        // Chunks
        .add_plugins(ChunkPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(PlayerPlugin)
        .add_plugins(SunPlugin)
        .run();
}
