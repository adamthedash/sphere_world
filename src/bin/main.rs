use bevy::{
    input::common_conditions::input_just_pressed,
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
};
use sphere_world::{
    assets::load_assets, chunks::ChunkPlugin, player::PlayerPlugin, sun::SunPlugin,
};

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
        .add_plugins(PlayerPlugin)
        .add_plugins(SunPlugin)
        .run();
}
