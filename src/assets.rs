use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

#[derive(Resource)]
pub struct AssetHandles {
    // Images
    pub uv_debug: Handle<Image>,
    pub hue_wheel: Handle<Image>,

    // Materials
    pub hue_material: Handle<StandardMaterial>,
}

pub fn load_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Images
    let uv_debug = asset_server.add(uv_debug_texture());
    let hue_wheel = asset_server.load("hue_wheel.png");

    // Materials
    let hue_material = asset_server.add(StandardMaterial {
        base_color_texture: Some(hue_wheel.clone()),
        ..default()
    });

    commands.insert_resource(AssetHandles {
        uv_debug,
        hue_wheel,
        hue_material,
    });
}

/// Creates a colorful test pattern
pub fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 16;

    let mut texture_data = Vec::with_capacity(TEXTURE_SIZE * TEXTURE_SIZE * 4);
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            texture_data.extend([
                (255. * y as f32 / (TEXTURE_SIZE as f32)).clamp(0., 255.) as u8, // R
                0,                                                               // G
                (255. * x as f32 / (TEXTURE_SIZE as f32)).clamp(0., 255.) as u8, // R
                255,                                                             // A
            ]);
        }
    }

    Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

pub fn hue_wheel_texture() -> Image {
    let bytes = include_bytes!("../assets/hue_wheel.png").to_vec();

    Image::new(
        Extent3d {
            width: bytes.len().isqrt() as u32,
            height: bytes.len().isqrt() as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}
