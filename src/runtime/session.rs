//! Session manager: load ONNX models with EP priority chain.

use super::ep::{resolve_ep_chain, EpPreference};
use ort::session::Session;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("model not found: {0}")]
    NotFound(PathBuf),
    #[error("ort: {0}")]
    Ort(#[from] ort::Error),
    #[error("{0}")]
    Other(String),
}

pub struct SessionManager {
    sessions: HashMap<String, Session>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Load (or return cached) a session for `key` from `model_path` using `ep_prefs`.
    pub fn load(
        &mut self,
        key: &str,
        model_path: &Path,
        ep_prefs: &[EpPreference],
    ) -> Result<&Session, SessionError> {
        if self.sessions.contains_key(key) {
            return Ok(self.sessions.get(key).unwrap());
        }
        if !model_path.exists() {
            return Err(SessionError::NotFound(model_path.to_path_buf()));
        }

        let eps = resolve_ep_chain(ep_prefs);
        info!(
            key,
            path = %model_path.display(),
            ep_count = eps.len(),
            "loading ONNX session"
        );

        let session = Session::builder()?
            .with_execution_providers(eps)?
            .commit_from_file(model_path)?;

        self.sessions.insert(key.to_string(), session);
        Ok(self.sessions.get(key).unwrap())
    }

    pub fn get(&self, key: &str) -> Option<&Session> {
        self.sessions.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Session> {
        self.sessions.get_mut(key)
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}
