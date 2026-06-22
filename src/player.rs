use bevy::prelude::*;
use bevy_enhanced_input::{actions, prelude::*};
use itertools::Itertools;
use rand::{RngExt, SeedableRng};

use crate::assets::AssetHandles;

#[derive(Component)]
pub struct Player;

/// Random point on unit sphere
fn random_unit_vector() -> Vec3 {
    let rng = rand::rngs::SmallRng::seed_from_u64(42);

    let pos = rng.random_iter::<f32>().take(3).collect_array().unwrap();
    Vec3::from_array(pos).normalize()
}

fn init_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetHandles>,
) {
    // Player spawns randomly on world surface
    // "Up" for the player is always away from planet centre
    let player_transform = {
        let up = random_unit_vector();
        let pos = up * 1.5;
        let forward = up.any_orthonormal_vector();

        Transform::from_translation(pos).looking_to(forward, up)
    };

    let shape = Capsule3d::new(0.01, 0.03);

    let mut player = commands.spawn((
        Player, //
        player_transform,
        Mesh3d(meshes.add(shape)),
        MeshMaterial3d(assets.hue_material.clone()),
    ));

    // Camera - 3rd person perspective
    player.with_children(|s| {
        let camera_transform =
            Transform::from_translation(Vec3::new(0., 0.2, 0.3)).looking_at(Vec3::ZERO, Vec3::Y);

        s.spawn((
            Camera3d::default(), //
            camera_transform,
        ));
    });
}

fn init_controller(mut commands: Commands, player: Single<Entity, With<Player>>) {
    commands.entity(*player).insert((
        WalkingCam, //
        actions!(
            WalkingCam[
            (
                Action::<Movement>::new(), //
                DeltaScale::default(),
                Bindings::spawn(Cardinal::arrows())
            ),
            (
                Action::<Looking>::new(), //
                DeltaScale::default(),
                Bindings::spawn(Spawn(Binding::mouse_motion())),
            )
            ]
        ),
    ));
}

/// Marker for when we're controlling the player in a normal walking around on the ground mode
#[derive(Component)]
struct WalkingCam;

#[derive(InputAction)]
#[action_output(Vec2)]
struct Movement;

fn move_player(event: On<Fire<Movement>>, mut player: Single<&mut Transform, With<Player>>) {
    let rot = Quat::from_axis_angle(*player.forward(), event.value.x)
        * Quat::from_axis_angle(*player.left(), event.value.y);

    player.rotate_around(Vec3::ZERO, rot);
}

#[derive(InputAction)]
#[action_output(Vec2)]
struct Looking;

fn rotate_player(event: On<Fire<Looking>>, mut player: Single<&mut Transform, With<Player>>) {
    let rot = Quat::from_axis_angle(*player.up(), event.value.x);
    // Normalise needed here because for whatever reason there's floating point error creeping in
    let rot = rot.normalize();
    player.rotate(rot);
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EnhancedInputPlugin)
            .add_systems(Startup, (init_player, init_controller).chain())
            .add_input_context::<WalkingCam>()
            .add_observer(move_player)
            .add_observer(rotate_player);
    }
}
