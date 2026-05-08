#![feature(file_buffered)]
#![feature(iter_map_windows)]
#![feature(iter_array_chunks)]
pub mod chunks;
pub mod drag_value;
pub mod noise;
pub mod polar;

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

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
