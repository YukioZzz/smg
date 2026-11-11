use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{
    common::{invalid, GenerationRequest, UsageInfo},
    validated::Normalizable,
};

// ============================================================================
// Embedding API
// ============================================================================

/// The most inputs one request carries: the public API's bound on an array
/// of strings or of token id arrays.
pub const MAX_INPUTS: usize = 2048;

/// The body of `/v1/embeddings`. Its rules are the public API's: `input` in
/// one of the four shapes `validate_input` names, `encoding_format` one of
/// the two the API defines, `dimensions` at least one. The validated
/// extractor answers a body outside them in the gateway's error envelope
/// before any worker sees it, on every path.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
pub struct EmbeddingRequest {
    /// ID of the model to use
    pub model: String,

    /// A string, an array of strings, an array of token ids, or an array of
    /// token id arrays, one per input; nothing empty, at most [`MAX_INPUTS`]
    /// inputs.
    #[validate(custom(function = "validate_input"))]
    pub input: Value,

    /// Optional encoding format: `float` or `base64`
    #[validate(custom(function = "validate_encoding_format"))]
    pub encoding_format: Option<String>,

    /// Optional user identifier
    pub user: Option<String>,

    /// Optional number of dimensions for the embedding, at least one
    #[validate(range(min = 1))]
    pub dimensions: Option<u32>,

    /// Engine extension: request id for tracking
    pub rid: Option<String>,
}

impl Normalizable for EmbeddingRequest {}

/// The shapes the public API takes for an `input`, on the classify route as
/// on embeddings: a string, an array of strings, an array of token ids
/// (non-negative integers), or an array of token id arrays. No string and no
/// array is empty, and an array of inputs has at most [`MAX_INPUTS`] members;
/// a flat array of token ids is one input, so that bound does not read its
/// length. The array's first member sets its kind, and a member of another
/// kind is refused by its index. The codes are the envelope's existing ones:
/// `empty_array` and `empty_input` for the empty shapes,
/// `array_above_max_length` over the bound, `invalid_type` for every other
/// shape.
pub(crate) fn validate_input(input: &Value) -> Result<(), validator::ValidationError> {
    const SHAPES: &str =
        "a string, an array of strings, an array of token ids or an array of token id arrays";
    let members = match input {
        Value::String(text) if text.is_empty() => {
            return Err(invalid(
                "empty_input",
                "Invalid 'input': empty string. Expected a string with minimum length 1, but got an empty string instead.",
            ));
        }
        Value::String(_) => return Ok(()),
        Value::Array(members) => members,
        _ => {
            return Err(invalid(
                "invalid_type",
                format!("Invalid 'input': expected {SHAPES}."),
            ));
        }
    };
    let Some(first) = members.first() else {
        return Err(invalid(
            "empty_array",
            "Invalid 'input': empty array. Expected an array with minimum length 1, but got an empty array instead.",
        ));
    };
    match first {
        Value::String(_) => {
            at_most_max_inputs(members)?;
            for (index, member) in members.iter().enumerate() {
                match member {
                    Value::String(text) if text.is_empty() => {
                        return Err(invalid(
                            "empty_input",
                            format!("Invalid 'input[{index}]': empty string. Expected a string with minimum length 1, but got an empty string instead."),
                        ));
                    }
                    Value::String(_) => {}
                    _ => {
                        return Err(invalid(
                            "invalid_type",
                            format!("Invalid 'input[{index}]': expected a string, as the array's first member is."),
                        ));
                    }
                }
            }
            Ok(())
        }
        Value::Number(_) => {
            if let Some(index) = members.iter().position(|member| member.as_u64().is_none()) {
                return Err(invalid(
                    "invalid_type",
                    format!("Invalid 'input[{index}]': expected a token id (a non-negative integer), as the array's first member is."),
                ));
            }
            Ok(())
        }
        Value::Array(_) => {
            at_most_max_inputs(members)?;
            for (index, member) in members.iter().enumerate() {
                let Value::Array(ids) = member else {
                    return Err(invalid(
                        "invalid_type",
                        format!("Invalid 'input[{index}]': expected an array of token ids, as the array's first member is."),
                    ));
                };
                if ids.is_empty() {
                    return Err(invalid(
                        "empty_array",
                        format!("Invalid 'input[{index}]': empty array. Expected an array of token ids with minimum length 1, but got an empty array instead."),
                    ));
                }
                if let Some(position) = ids.iter().position(|id| id.as_u64().is_none()) {
                    return Err(invalid(
                        "invalid_type",
                        format!("Invalid 'input[{index}][{position}]': expected a token id (a non-negative integer)."),
                    ));
                }
            }
            Ok(())
        }
        _ => Err(invalid(
            "invalid_type",
            format!("Invalid 'input[0]': expected {SHAPES}."),
        )),
    }
}

/// An array of inputs within the public API's bound.
fn at_most_max_inputs(inputs: &[Value]) -> Result<(), validator::ValidationError> {
    if inputs.len() > MAX_INPUTS {
        return Err(invalid(
            "array_above_max_length",
            format!(
                "Invalid 'input': {} inputs. Expected an array with maximum length {MAX_INPUTS}.",
                inputs.len()
            ),
        ));
    }
    Ok(())
}

/// `encoding_format` is one of the two the public API defines.
fn validate_encoding_format(format: &str) -> Result<(), validator::ValidationError> {
    match format {
        "float" | "base64" => Ok(()),
        other => Err(invalid(
            "unsupported_value",
            format!(
                "Invalid value: '{other}' for parameter 'encoding_format'; supported values are \
                 'float' and 'base64'"
            ),
        )),
    }
}

impl GenerationRequest for EmbeddingRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        // Embeddings are non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Best effort: extract text content for routing decisions
        match &self.input {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingObject {
    pub object: String, // "embedding"
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingResponse {
    pub object: String, // "list"
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: UsageInfo,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::classify::ClassifyRequest;

    fn request(value: Value) -> EmbeddingRequest {
        serde_json::from_value(value).expect("an embeddings request")
    }

    /// The code of the first violation the rules file under `field`.
    fn code_at(errors: &validator::ValidationErrors, field: &str) -> String {
        let field_errors = errors.field_errors();
        let found = field_errors
            .get(field)
            .unwrap_or_else(|| panic!("a violation under '{field}': {errors:?}"));
        found[0].code.to_string()
    }

    /// The four shapes the public API takes pass, with the bound's last
    /// member, a `dimensions` of one and both encoding formats.
    #[test]
    fn the_shapes_the_public_api_takes_pass() {
        let many: Vec<Value> = vec![json!("x"); MAX_INPUTS];
        for input in [
            json!("hello"),
            json!(["a", "b"]),
            json!([1, 2, 3]),
            json!([[1, 2], [3]]),
            Value::Array(many),
        ] {
            let within = request(json!({"model": "m", "input": input}));
            assert!(within.validate().is_ok(), "{}", within.input);
        }
        for options in [
            json!({"dimensions": 1, "encoding_format": "float"}),
            json!({"encoding_format": "base64"}),
        ] {
            let mut body = json!({"model": "m", "input": "hello"});
            body.as_object_mut()
                .expect("an object")
                .extend(options.as_object().expect("an object").clone());
            assert!(request(body.clone()).validate().is_ok(), "{body}");
        }
    }

    /// Every shape the public API refuses is refused under `input` with the
    /// envelope's existing code for it.
    #[test]
    fn the_shapes_the_public_api_refuses_are_refused_under_input() {
        let over: Vec<Value> = vec![json!("x"); MAX_INPUTS + 1];
        for (input, code) in [
            (json!([]), "empty_array"),
            (json!(""), "empty_input"),
            (json!(["a", ""]), "empty_input"),
            (json!({"text": "x"}), "invalid_type"),
            (json!(null), "invalid_type"),
            (json!(42), "invalid_type"),
            (json!(true), "invalid_type"),
            (json!(["a", 1, null]), "invalid_type"),
            (json!([1, "a"]), "invalid_type"),
            (json!([1, -2]), "invalid_type"),
            (json!([[1, 2], "a"]), "invalid_type"),
            (json!([[1, 2], []]), "empty_array"),
            (json!([[1, 2.5]]), "invalid_type"),
            (json!([null]), "invalid_type"),
            (Value::Array(over), "array_above_max_length"),
        ] {
            let refused = request(json!({"model": "m", "input": input}));
            let errors = refused
                .validate()
                .expect_err("a shape the public API refuses is refused");
            assert_eq!(code_at(&errors, "input"), code, "{}", refused.input);
        }
    }

    /// A flat array of token ids is one input: the bound on inputs does not
    /// read its length.
    #[test]
    fn a_flat_array_of_token_ids_is_one_input() {
        let ids: Vec<Value> = (0..MAX_INPUTS as u64 + 1).map(Value::from).collect();
        let request = request(json!({"model": "m", "input": Value::Array(ids)}));
        assert!(request.validate().is_ok());
    }

    /// `dimensions` under one and an `encoding_format` the API does not
    /// define are refused under their own fields.
    #[test]
    fn the_options_are_bounded_as_the_public_api_bounds_them() {
        let zero = request(json!({"model": "m", "input": "hello", "dimensions": 0}));
        let errors = zero.validate().expect_err("dimensions 0 is refused");
        assert_eq!(code_at(&errors, "dimensions"), "range");

        let hex = request(json!({"model": "m", "input": "hello", "encoding_format": "hex"}));
        let errors = hex.validate().expect_err("an unknown format is refused");
        assert_eq!(code_at(&errors, "encoding_format"), "unsupported_value");
    }

    /// The classify route runs the same `input` rule.
    #[test]
    fn the_classify_request_runs_the_same_input_rule() {
        let classify = |input: Value| -> ClassifyRequest {
            serde_json::from_value(json!({"model": "m", "input": input}))
                .expect("a classify request")
        };
        for served in [json!("hello"), json!(["a", "b"]), json!([1, 2, 3])] {
            assert!(classify(served.clone()).validate().is_ok(), "{served}");
        }
        for (input, code) in [
            (json!([]), "empty_array"),
            (json!(""), "empty_input"),
            (json!({"a": 1}), "invalid_type"),
            (json!(["a", 1]), "invalid_type"),
        ] {
            let errors = classify(input.clone())
                .validate()
                .expect_err("a shape the public API refuses is refused");
            assert_eq!(code_at(&errors, "input"), code, "{input}");
        }
    }
}
