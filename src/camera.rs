use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
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

fn zoom_camera(
    mut camera: Single<&mut Transform, With<Camera3d>>,
    time: Res<Time>,
    mouse: Res<AccumulatedMouseScroll>,
) {
    const SPEED: f32 = 0.1;

    let direction = camera.translation * -1.;
    let displacement = direction * mouse.delta.y * time.delta_secs() * SPEED;
    camera.translation += displacement;
}

fn init_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4., 4.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
    ));
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_camera)
            .add_systems(Update, (drag_camera, zoom_camera));
    }
}
