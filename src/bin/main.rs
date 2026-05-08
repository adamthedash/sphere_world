use std::path::Path;

use bevy::{
    ecs::error::Result,
    input::mouse::AccumulatedMouseMotion,
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
};
use bevy_egui::{egui::Slider, prelude::*};
use glam::Vec3A;
use hexasphere::shapes::IcoSphere;
use sphere_world::{
    assets::load_assets,
    chunks::ChunkPlugin,
    noise::{NoiseChanged, NoiseConfig, NoiseConfigWidget},
    polar::PolarCoord,
};

fn drag_camera(
    mut camera: Single<&mut Transform, With<Camera3d>>,
    time: Res<Time>,
    mouse: Res<AccumulatedMouseMotion>,
) {
    const SPEED: f32 = 0.3;
    let r1 = Quat::from_axis_angle(
        camera.right().as_vec3().normalize(),
        -mouse.delta.y * time.delta_secs() * SPEED,
    );
    let r2 = Quat::from_axis_angle(
        camera.up().as_vec3().normalize(),
        -mouse.delta.x * time.delta_secs() * SPEED,
    );
    let rot = r2.mul_quat(r1);
    camera.rotate_around(Vec3::ZERO, rot);
}

struct SphereData {
    origin: Vec3A,
    polar: PolarCoord,
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
    let mesh = meshes.get_mut(mesh.0.id()).expect("mesh not found");

    // Recompute height map & normals based with new noise function
    let noise = noise_config.generator();

    let points = planet
        .0
        .raw_data()
        .iter()
        .map(|d| {
            // let height = noise.get(d.origin.to_array().map(|x| x as f64));
            // let point = d.origin * (1. + height as f32);
            // point.to_array()
            d.origin.to_array()
        })
        .collect::<Vec<_>>();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, points);
    mesh.compute_normals();

    should_regen.0 = false;
}

fn setup(mut commands: Commands) {
    // Light
    commands.spawn((
        PointLight {
            // shadow_maps_enabled: true,
            intensity: 10_000_000.,
            range: 100.0,
            shadow_depth_bias: 0.2,
            ..default()
        },
        Transform::from_xyz(10., 10., 10.),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4., 4.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
    ));
}

#[derive(Resource)]
struct SphereConfig {
    num_subdivisions: usize,
}

impl SphereConfig {
    pub fn generate_sphere(&self) -> Planet {
        todo!()
    }
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
        .add_systems(Startup, |mut c: ResMut<WireframeConfig>| c.global = true)
        // UI
        .add_plugins(EguiPlugin::default())
        .add_systems(EguiPrimaryContextPass, draw_ui)
        .insert_resource(ClearColor(Color::BLACK))
        // Assets
        .add_systems(PreStartup, load_assets)
        // Chunks
        .add_plugins(ChunkPlugin)
        //
        .insert_resource(NoiseConfig::default())
        .add_systems(Startup, setup)
        .insert_resource(ShouldRegenerateMesh(true))
        .add_message::<NoiseChanged>()
        .add_systems(Update, drag_camera)
        .run();
}
