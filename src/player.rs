use std::f32::consts::PI;

use bevy::prelude::*;

use crate::assets::AssetHandles;

#[derive(Component)]
pub struct Player;

fn init_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetHandles>,
) {
    let mesh = Sphere::new(0.1).mesh().ico(3).unwrap();
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

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_player)
            .add_systems(Update, move_player);
    }
}
