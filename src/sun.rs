use std::f32::consts::PI;

use bevy::{
    color::palettes::css::{RED, YELLOW},
    prelude::*,
};

#[derive(Component)]
pub struct Sun;

pub fn init_sun(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Sun - point light
    let transform = Transform::from_xyz(3., 3., 3.);
    commands.spawn((
        PointLight {
            intensity: 100_000.,
            shadow_maps_enabled: true,
            ..default()
        },
        transform,
        Sun,
    ));
    let mesh = Sphere::new(0.1).mesh().uv(32, 18);
    commands.spawn((
        transform,
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial::from_color(YELLOW))),
        Sun,
    ));

    // Ambient light
    commands.insert_resource(GlobalAmbientLight {
        color: RED.into(),
        brightness: 10.,
        ..default()
    });
}

fn move_sun(sun: Query<&mut Transform, With<Sun>>, time: Res<Time>) {
    let rot = Quat::from_rotation_y(PI * 0.1 * time.delta_secs());

    for mut transform in sun {
        transform.rotate_around(Vec3::ZERO, rot);
    }
}

pub struct SunPlugin;

impl Plugin for SunPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_sun)
            .add_systems(Update, move_sun);
    }
}
