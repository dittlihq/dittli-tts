use js_sys::{BigInt64Array, Float32Array, Object, Reflect};
use rten::{Model, NodeId, Value};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct DittliSession {
    model: Model,
    /// (name, node_id) for each model input, in graph order.
    inputs: Vec<(String, NodeId)>,
    output_id: NodeId,
}

#[wasm_bindgen]
impl DittliSession {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<DittliSession, JsValue> {
        let model = Model::load(model_bytes.to_vec())
            .map_err(|e| JsValue::from_str(&format!("model load: {e}")))?;

        let inputs: Vec<(String, NodeId)> = model
            .input_ids()
            .iter()
            .filter_map(|&id| {
                let name = model.node_info(id)?.name()?.to_string();
                Some((name, id))
            })
            .collect();

        let output_id = model
            .output_ids()
            .first()
            .copied()
            .ok_or_else(|| JsValue::from_str("model has no outputs"))?;

        Ok(DittliSession { model, inputs, output_id })
    }

    pub fn run(&self, feeds: JsValue) -> Result<Float32Array, JsValue> {
        let feeds_obj = feeds
            .dyn_ref::<Object>()
            .ok_or_else(|| JsValue::from_str("feeds must be an object"))?;

        let mut run_inputs: Vec<(NodeId, rten::ValueOrView<'_>)> = Vec::new();

        for (name, id) in &self.inputs {
            let entry = Reflect::get(feeds_obj, &JsValue::from_str(name))
                .map_err(|_| JsValue::from_str(&format!("missing feed: {name}")))?;
            let entry_obj = entry
                .dyn_ref::<Object>()
                .ok_or_else(|| JsValue::from_str(&format!("feed '{name}' not an object")))?;

            let dtype = Reflect::get(entry_obj, &JsValue::from_str("type"))
                .ok()
                .and_then(|v| v.as_string())
                .ok_or_else(|| JsValue::from_str(&format!("feed '{name}'.type missing")))?;

            let shape_val = Reflect::get(entry_obj, &JsValue::from_str("shape"))
                .map_err(|_| JsValue::from_str(&format!("feed '{name}' missing shape")))?;
            let shape = js_array_to_usize_vec(&shape_val, name)?;

            let data_val = Reflect::get(entry_obj, &JsValue::from_str("data"))
                .map_err(|_| JsValue::from_str(&format!("feed '{name}' missing data")))?;

            let value: Value = match dtype.as_str() {
                "int64" => {
                    // ONNX int64 → rten i32 (phoneme indices always fit in i32)
                    let arr = BigInt64Array::from(data_val);
                    let mut buf_i64 = vec![0i64; arr.length() as usize];
                    arr.copy_to(&mut buf_i64);
                    let buf_i32: Vec<i32> = buf_i64.iter().map(|&v| v as i32).collect();
                    Value::from_shape(shape, buf_i32)
                        .map_err(|e| JsValue::from_str(&format!("tensor '{name}': {e}")))?
                }
                "float32" => {
                    let arr = Float32Array::from(data_val);
                    let mut buf = vec![0f32; arr.length() as usize];
                    arr.copy_to(&mut buf);
                    Value::from_shape(shape, buf)
                        .map_err(|e| JsValue::from_str(&format!("tensor '{name}': {e}")))?
                }
                other => {
                    return Err(JsValue::from_str(&format!(
                        "unsupported dtype '{other}' for '{name}'"
                    )))
                }
            };

            run_inputs.push((*id, value.into()));
        }

        let mut outputs = self
            .model
            .run(run_inputs, &[self.output_id], None)
            .map_err(|e| JsValue::from_str(&format!("inference: {e}")))?;

        let output = outputs.remove(0);
        let tensor_view = output
            .as_tensor_view::<f32>()
            .ok_or_else(|| JsValue::from_str("output is not a float32 tensor"))?;
        let samples = tensor_view
            .data()
            .ok_or_else(|| JsValue::from_str("output tensor is not contiguous"))?;

        let result = Float32Array::new_with_length(samples.len() as u32);
        result.copy_from(samples);
        Ok(result)
    }

    pub fn release(&self) {}
}

fn js_array_to_usize_vec(val: &JsValue, name: &str) -> Result<Vec<usize>, JsValue> {
    let arr = js_sys::Array::from(val);
    (0..arr.length())
        .map(|i| {
            arr.get(i)
                .as_f64()
                .map(|v| v as usize)
                .ok_or_else(|| {
                    JsValue::from_str(&format!("shape element for '{name}' is not a number"))
                })
        })
        .collect()
}
