//! Supported language definitions, ISO/BCP-47 codes, and script metadata.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SupportedLanguage {
    #[default]
    Auto,
    En,
    Fr,
    Es,
    Ar,
    Ru,
    Hi,
    Zh,
    ZhTw,
    Ko,
    Ja,
    Ur,
    Fa,
    Tr,
    Az,
    Hy,
}

impl SupportedLanguage {
    /// Complete array of all supported language choices.
    pub const ALL: &'static [SupportedLanguage] = &[
        Self::Auto,
        Self::En,
        Self::Fr,
        Self::Es,
        Self::Ar,
        Self::Ru,
        Self::Hi,
        Self::Zh,
        Self::ZhTw,
        Self::Ko,
        Self::Ja,
        Self::Ur,
        Self::Fa,
        Self::Tr,
        Self::Az,
        Self::Hy,
    ];

    /// Standard BCP-47 / Whisper model language code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::En => "en",
            Self::Fr => "fr",
            Self::Es => "es",
            Self::Ar => "ar",
            Self::Ru => "ru",
            Self::Hi => "hi",
            // Whisper has no separate Taiwanese/Hokkien code; both map to zh.
            Self::Zh | Self::ZhTw => "zh",
            Self::Ko => "ko",
            Self::Ja => "ja",
            Self::Ur => "ur",
            Self::Fa => "fa",
            Self::Tr => "tr",
            Self::Az => "az",
            Self::Hy => "hy",
        }
    }

    /// User-friendly display label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::En => "English",
            Self::Fr => "French",
            Self::Es => "Spanish",
            Self::Ar => "Arabic",
            Self::Ru => "Russian",
            Self::Hi => "Hindi",
            Self::Zh => "Mandarin (zh)",
            Self::ZhTw => "Taiwanese (zh)",
            Self::Ko => "Korean",
            Self::Ja => "Japanese",
            Self::Ur => "Urdu (PK)",
            Self::Fa => "Farsi",
            Self::Tr => "Turkish",
            Self::Az => "Azerbaijani",
            Self::Hy => "Armenian",
        }
    }

    /// True if the language uses a Right-to-Left (RTL) script.
    pub fn is_rtl(&self) -> bool {
        matches!(self, Self::Ar | Self::Ur | Self::Fa)
    }

    /// True if the language uses a Chinese, Japanese, or Korean (CJK) script.
    pub fn is_cjk(&self) -> bool {
        matches!(self, Self::Zh | Self::ZhTw | Self::Ko | Self::Ja)
    }

    /// Parse a language code case-insensitively (e.g. "en", "FR", "zh-tw", "auto").
    pub fn from_code(code: &str) -> Option<Self> {
        let trimmed = code.trim().to_ascii_lowercase();
        match trimmed.as_str() {
            "auto" | "detect" | "multilingual" => Some(Self::Auto),
            "en" => Some(Self::En),
            "fr" => Some(Self::Fr),
            "es" => Some(Self::Es),
            "ar" => Some(Self::Ar),
            "ru" => Some(Self::Ru),
            "hi" => Some(Self::Hi),
            "zh-tw" | "zhtw" | "zh_tw" => Some(Self::ZhTw),
            "zh" | "zh-cn" | "zh_cn" => Some(Self::Zh),
            "ko" => Some(Self::Ko),
            "ja" => Some(Self::Ja),
            "ur" => Some(Self::Ur),
            "fa" => Some(Self::Fa),
            "tr" => Some(Self::Tr),
            "az" => Some(Self::Az),
            "hy" => Some(Self::Hy),
            _ => None,
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_languages_count() {
        assert_eq!(SupportedLanguage::ALL.len(), 16);
    }

    #[test]
    fn test_codes_and_labels_non_empty() {
        for lang in SupportedLanguage::ALL {
            assert!(!lang.code().is_empty());
            assert!(!lang.label().is_empty());
        }
    }

    #[test]
    fn test_rtl_classification() {
        assert!(SupportedLanguage::Ar.is_rtl());
        assert!(SupportedLanguage::Ur.is_rtl());
        assert!(SupportedLanguage::Fa.is_rtl());
        assert!(!SupportedLanguage::En.is_rtl());
        assert!(!SupportedLanguage::Zh.is_rtl());
    }

    #[test]
    fn test_cjk_classification() {
        assert!(SupportedLanguage::Zh.is_cjk());
        assert!(SupportedLanguage::ZhTw.is_cjk());
        assert!(SupportedLanguage::Ko.is_cjk());
        assert!(SupportedLanguage::Ja.is_cjk());
        assert!(!SupportedLanguage::En.is_cjk());
        assert!(!SupportedLanguage::Ar.is_cjk());
    }

    #[test]
    fn test_from_code_parsing() {
        assert_eq!(SupportedLanguage::from_code("en"), Some(SupportedLanguage::En));
        assert_eq!(SupportedLanguage::from_code("EN"), Some(SupportedLanguage::En));
        assert_eq!(SupportedLanguage::from_code("fr"), Some(SupportedLanguage::Fr));
        assert_eq!(SupportedLanguage::from_code("es"), Some(SupportedLanguage::Es));
        assert_eq!(SupportedLanguage::from_code("ar"), Some(SupportedLanguage::Ar));
        assert_eq!(SupportedLanguage::from_code("ru"), Some(SupportedLanguage::Ru));
        assert_eq!(SupportedLanguage::from_code("hi"), Some(SupportedLanguage::Hi));
        assert_eq!(SupportedLanguage::from_code("zh"), Some(SupportedLanguage::Zh));
        assert_eq!(SupportedLanguage::from_code("zh-tw"), Some(SupportedLanguage::ZhTw));
        assert_eq!(SupportedLanguage::from_code("ko"), Some(SupportedLanguage::Ko));
        assert_eq!(SupportedLanguage::from_code("ja"), Some(SupportedLanguage::Ja));
        assert_eq!(SupportedLanguage::from_code("ur"), Some(SupportedLanguage::Ur));
        assert_eq!(SupportedLanguage::from_code("fa"), Some(SupportedLanguage::Fa));
        assert_eq!(SupportedLanguage::from_code("tr"), Some(SupportedLanguage::Tr));
        assert_eq!(SupportedLanguage::from_code("az"), Some(SupportedLanguage::Az));
        assert_eq!(SupportedLanguage::from_code("hy"), Some(SupportedLanguage::Hy));
        assert_eq!(SupportedLanguage::from_code("auto"), Some(SupportedLanguage::Auto));
        assert_eq!(SupportedLanguage::from_code("multilingual"), Some(SupportedLanguage::Auto));
        assert_eq!(SupportedLanguage::from_code("unknown_xyz"), None);
    }
}
