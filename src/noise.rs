use std::path::Path;

use bevy::prelude::*;
use bevy_egui::egui::{self, Widget};
use noise::{NoiseFn, ScaleBias, ScalePoint, Simplex};
use serde::{Deserialize, Serialize};

use crate::drag_value::LogDragValue;

pub struct AddMany<S> {
    sources: Vec<S>,
}

impl<S> AddMany<S> {
    pub fn new(sources: Vec<S>) -> Self {
        Self { sources }
    }
}

impl<T, S, const DIM: usize> NoiseFn<T, DIM> for AddMany<S>
where
    S: NoiseFn<T, DIM>,
    T: Copy,
{
    fn get(&self, point: [T; DIM]) -> f64 {
        self.sources.iter().map(|s| s.get(point)).sum()
    }
}

#[derive(Resource, Serialize, Deserialize)]
pub struct NoiseConfig {
    pub octaves: Vec<NoiseOctave>,
    pub input_scale: f64,
    pub output_scale: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            octaves: vec![NoiseOctave::default()],
            input_scale: 1.,
            output_scale: 1.,
        }
    }
}

impl NoiseConfig {
    pub fn save(&self, path: &Path) -> Result {
        let dir = path.parent().ok_or("No parent")?;
        std::fs::create_dir_all(dir)?;

        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;

        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open_buffered(path)?;
        let config = serde_json::from_reader(file)?;

        Ok(config)
    }
}

impl NoiseConfig {
    pub fn generator(&self) -> impl NoiseFn<f64, 3> {
        const SEED: u32 = 42;

        let simplex = Simplex::new(SEED);

        let total_octave_scale = self.octaves.iter().map(|o| o.output_scale).sum::<f64>();

        // Per-octave settings
        let octaves = self
            .octaves
            .iter()
            .map(|o| {
                ScalePoint::new(
                    ScaleBias::new(simplex).set_scale(o.output_scale / total_octave_scale),
                )
                .set_scale(o.input_scale)
            })
            .collect();
        let noise = AddMany::new(octaves);

        // Global settings
        ScaleBias::new(ScalePoint::new(noise).set_scale(self.output_scale))
            .set_scale(self.input_scale)
    }
}

#[derive(Serialize, Deserialize)]
pub struct NoiseOctave {
    pub output_scale: f64,
    pub input_scale: f64,
}

impl Default for NoiseOctave {
    fn default() -> Self {
        Self {
            output_scale: 1.,
            input_scale: 1.,
        }
    }
}

#[derive(Message, Default)]
pub struct NoiseChanged;

pub struct NoiseConfigWidget<'a, 'w> {
    config: &'a mut NoiseConfig,
    notify_changed: &'a mut MessageWriter<'w, NoiseChanged>,
}

impl<'a, 'w> NoiseConfigWidget<'a, 'w> {
    pub fn new(
        config: &'a mut NoiseConfig,
        notify_changed: &'a mut MessageWriter<'w, NoiseChanged>,
    ) -> Self {
        Self {
            config,
            notify_changed,
        }
    }
}

impl Widget for NoiseConfigWidget<'_, '_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.vertical(|ui| {
            // Octaves
            let mut num_octaves = self.config.octaves.len();
            ui.add(egui::Slider::new(&mut num_octaves, 1..=8).text("# Octaves"));

            if num_octaves != self.config.octaves.len() {
                while num_octaves < self.config.octaves.len() {
                    self.config.octaves.pop();
                }
                while num_octaves > self.config.octaves.len() {
                    self.config.octaves.push(NoiseOctave::default());
                }

                self.notify_changed.write_default();
            }

            // Individual octaves
            for octave in &mut self.config.octaves {
                // Amplitude
                ui.add(
                    LogDragValue::new(&mut octave.output_scale)
                        .label("Scale output")
                        .on_change(|| {
                            self.notify_changed.write_default();
                        }),
                );

                // Frequency
                ui.add(
                    LogDragValue::new(&mut octave.input_scale)
                        .label("Scale input")
                        .on_change(|| {
                            self.notify_changed.write_default();
                        }),
                );

                ui.separator();
            }
        })
        .response
    }
}
