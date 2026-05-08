use bevy_egui::egui::{DragValue, Response, Slider, Ui, Widget};

/// Log-scaled drag value. Only updates source value when it has changed
pub struct LogDragValue<'a, F> {
    source_ref: &'a mut f64,
    label: Option<&'a str>,
    on_change: Option<F>,
}

impl<'a, F> LogDragValue<'a, F> {
    pub fn new(value: &'a mut f64) -> Self {
        Self {
            source_ref: value,
            label: None,
            on_change: None,
        }
    }

    pub fn label(self, label: &'a str) -> Self {
        Self {
            label: Some(label),
            ..self
        }
    }

    pub fn on_change(self, func: F) -> Self {
        Self {
            on_change: Some(func),
            ..self
        }
    }
}

impl<F> Widget for LogDragValue<'_, F>
where
    F: FnMut(),
{
    fn ui(self, ui: &mut Ui) -> Response {
        ui.horizontal(|ui| {
            let previous = *self.source_ref;
            let mut log_val = previous.log2();
            ui.add(DragValue::new(&mut log_val).range(-10..=10).speed(0.01));
            let new = 2_f64.powf(log_val);

            if previous != new {
                *self.source_ref = new;

                if let Some(mut func) = self.on_change {
                    func()
                }
            }

            if let Some(label) = self.label {
                ui.label(label);
            }
        })
        .response
    }
}
