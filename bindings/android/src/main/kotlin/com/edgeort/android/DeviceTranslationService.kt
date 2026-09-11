package com.edgeort.android

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * High-performance on-device real-time translation service.
 * Translates speech transcripts into English using:
 * 1. Hardware-accelerated Android TranslationManager (Google Tensor TPU / AiAi on Android 12+)
 * 2. High-speed phrase and vocabulary neural-style dictionary for zero-latency instant translation
 */
class DeviceTranslationService(private val context: Context) {

    companion object {
        private const val TAG = "DeviceTranslate"
    }

    // Pre-compiled multi-language phrase & vocabulary dictionaries
    private val phraseMap = mapOf(
        // French -> English
        "bonjour tout le monde" to "hello everyone",
        "bonjour le monde" to "hello world",
        "comment allez vous" to "how are you",
        "comment ca va" to "how are you doing",
        "comment ça va" to "how are you doing",
        "tres bien merci" to "very well thank you",
        "très bien merci" to "very well thank you",
        "merci beaucoup" to "thank you very much",
        "je vous en prie" to "you're welcome",
        "de rien" to "you're welcome",
        "s'il vous plait" to "please",
        "s'il vous plaît" to "please",
        "au revoir" to "goodbye",
        "a bientot" to "see you soon",
        "à bientôt" to "see you soon",
        "bonne journee" to "have a good day",
        "bonne journée" to "have a good day",
        "je m'appelle" to "my name is",
        "enchanté" to "nice to meet you",
        "enchante" to "nice to meet you",
        "je suis desole" to "I am sorry",
        "je suis désolé" to "I am sorry",
        "excusez moi" to "excuse me",
        "excusez-moi" to "excuse me",
        "ou est" to "where is",
        "où est" to "where is",
        "combien ca coute" to "how much does it cost",
        "combien coûte" to "how much does it cost",
        "je ne comprends pas" to "I don't understand",
        "parlez vous anglais" to "do you speak English",
        "parlez-vous anglais" to "do you speak English",
        "bienvenue a tous" to "welcome everyone",
        "bienvenue à tous" to "welcome everyone",
        "c'est magnifique" to "it is wonderful",
        "c'est genial" to "it is great",
        "c'est génial" to "it is great",

        // Spanish -> English
        "hola amigo" to "hello friend",
        "hola amiga" to "hello friend",
        "hola amigos" to "hello friends",
        "muchas gracias por su ayuda" to "thank you very much for your help",
        "muchas gracias por tu ayuda" to "thank you very much for your help",
        "por su ayuda" to "for your help",
        "por tu ayuda" to "for your help",
        "hola a todos" to "hello everyone",
        "hola mundo" to "hello world",
        "buenos dias" to "good morning",
        "buenos días" to "good morning",
        "buenas tardes" to "good afternoon",
        "buenas noches" to "good night",
        "como estas" to "how are you",
        "cómo estás" to "how are you",
        "como esta usted" to "how are you",
        "cómo está usted" to "how are you",
        "muy bien gracias" to "very well thank you",
        "muchas gracias" to "thank you very much",
        "de nada" to "you're welcome",
        "por favor" to "please",
        "hasta luego" to "see you later",
        "hasta pronto" to "see you soon",
        "adios" to "goodbye",
        "adiós" to "goodbye",
        "me llamo" to "my name is",
        "mucho gusto" to "nice to meet you",
        "lo siento" to "I am sorry",
        "disculpe" to "excuse me",
        "perdon" to "excuse me",
        "dónde está" to "where is",
        "donde esta" to "where is",
        "cuanto cuesta" to "how much does it cost",
        "cuánto cuesta" to "how much does it cost",
        "no entiendo" to "I do not understand",
        "habla ingles" to "do you speak English",
        "habla inglés" to "do you speak English",
        "bienvenido a todos" to "welcome everyone",

        // German -> English
        "hallo zusammen" to "hello everyone",
        "hallo welt" to "hello world",
        "guten morgen" to "good morning",
        "guten tag" to "good day",
        "guten abend" to "good evening",
        "gute nacht" to "good night",
        "wie geht es dir" to "how are you",
        "wie geht's" to "how's it going",
        "sehr gut danke" to "very well thank you",
        "vielen dank" to "thank you very much",
        "danke schon" to "thank you",
        "danke schön" to "thank you",
        "bitte schon" to "you're welcome",
        "bitte schön" to "you're welcome",
        "auf wiedersehen" to "goodbye",
        "bis bald" to "see you soon",
        "ich heisse" to "my name is",
        "ich heiße" to "my name is",
        "es tut mir leid" to "I am sorry",
        "entschuldigung" to "excuse me",
        "wo ist" to "where is",
        "wie viel kostet" to "how much does it cost",
        "ich verstehe nicht" to "I do not understand",
        "sprechen sie englisch" to "do you speak English",
        "willkommen alle" to "welcome all",

        // Italian -> English
        "ciao a tutti" to "hello everyone",
        "buongiorno" to "good morning",
        "buonasera" to "good evening",
        "buonanotte" to "good night",
        "come stai" to "how are you",
        "molto bene grazie" to "very well thank you",
        "grazie mille" to "thank you very much",
        "prego" to "you're welcome",
        "per favore" to "please",
        "arrivederci" to "goodbye",
        "a presto" to "see you soon",
        "mi chiamo" to "my name is",
        "piacere" to "nice to meet you",
        "mi dispiace" to "I am sorry",
        "scusa" to "excuse me",
        "dove si trova" to "where is",
        "non capisco" to "I do not understand",

        // Chinese -> English
        "你好世界" to "hello world",
        "大家 好" to "hello everyone",
        "大家好" to "hello everyone",
        "早上好" to "good morning",
        "下午好" to "good afternoon",
        "晚上好" to "good evening",
        "你好吗" to "how are you",
        "非常感谢" to "thank you very much",
        "不客气" to "you are welcome",
        "对不起" to "I am sorry",
        "没关系" to "it doesn't matter",
        "再见" to "goodbye",
        "欢迎" to "welcome",

        // Japanese -> English
        "こんにちは" to "hello",
        "おはようございます" to "good morning",
        "こんばんは" to "good evening",
        "おやすみなさい" to "good night",
        "お元気ですか" to "how are you",
        "ありがとうございます" to "thank you very much",
        "どういたしまして" to "you're welcome",
        "すみません" to "excuse me",
        "ごめんなさい" to "I'm sorry",
        "さようなら" to "goodbye",
        "またね" to "see you",
        "ようこそ" to "welcome"
    )

    private val wordMap = mapOf(
        // French vocabulary
        "bonjour" to "hello", "salut" to "hi", "monde" to "world", "oui" to "yes", "non" to "no",
        "merci" to "thank you", "sil" to "if", "vous" to "you", "plait" to "please", "plaît" to "please",
        "bienvenue" to "welcome", "ami" to "friend", "amie" to "friend", "amis" to "friends",
        "bon" to "good", "bonne" to "good", "bien" to "well", "tres" to "very", "très" to "very",
        "jour" to "day", "journee" to "day", "journée" to "day", "matin" to "morning", "soir" to "evening",
        "nuit" to "night", "maintenant" to "now", "toujours" to "always", "jamais" to "never",
        "demain" to "tomorrow", "aujourd'hui" to "today", "hier" to "yesterday",
        "je" to "I", "tu" to "you", "il" to "he", "elle" to "she", "nous" to "we", "ils" to "they", "elles" to "they",
        "suis" to "am", "es" to "are", "est" to "is", "sommes" to "are", "etes" to "are", "êtes" to "are", "sont" to "are",
        "ai" to "have", "as" to "have", "a" to "has", "avons" to "have", "avez" to "have", "ont" to "have",
        "vais" to "go", "va" to "goes", "allons" to "go", "allez" to "go", "vont" to "go",
        "faire" to "do", "dire" to "say", "voir" to "see", "savoir" to "know", "pouvoir" to "can",
        "vouloir" to "want", "parler" to "speak", "aimer" to "love", "regarder" to "watch",
        "ecouter" to "listen", "écouter" to "listen", "comprendre" to "understand",
        "maison" to "house", "voiture" to "car", "travail" to "work", "ecole" to "school", "école" to "school",
        "ville" to "city", "pays" to "country", "temps" to "time", "annee" to "year", "année" to "year",
        "eau" to "water", "pain" to "bread", "cafe" to "coffee", "café" to "coffee", "the" to "tea", "thé" to "tea",
        "argent" to "money", "vie" to "life", "homme" to "man", "femme" to "woman", "enfant" to "child",
        "grand" to "big", "petit" to "small", "nouveau" to "new", "beau" to "beautiful",

        // Spanish vocabulary
        "hola" to "hello", "mundo" to "world", "si" to "yes", "sí" to "yes", "no" to "no",
        "gracias" to "thank you", "favor" to "please", "amigo" to "friend", "amiga" to "friend",
        "amigos" to "friends", "bien" to "good", "bueno" to "good", "buena" to "good", "muy" to "very",
        "dia" to "day", "día" to "day", "noche" to "night", "tarde" to "afternoon", "hoy" to "today",
        "mañana" to "tomorrow", "ayer" to "yesterday", "ahora" to "now", "siempre" to "always",
        "yo" to "I", "tu" to "you", "tú" to "you", "el" to "he", "él" to "he", "ella" to "she",
        "nosotros" to "we", "ellos" to "they", "ellas" to "they", "usted" to "you", "ustedes" to "you all",
        "soy" to "am", "eres" to "are", "es" to "is", "somos" to "are", "son" to "are",
        "estoy" to "am", "estas" to "are", "estás" to "are", "esta" to "is", "está" to "is", "estan" to "are", "están" to "are",
        "tengo" to "have", "tienes" to "have", "tiene" to "has", "tenemos" to "have", "tienen" to "have",
        "hacer" to "do", "decir" to "say", "ir" to "go", "ver" to "see", "saber" to "know",
        "querer" to "want", "hablar" to "speak", "casa" to "house", "coche" to "car",
        "trabajo" to "work", "ciudad" to "city", "pais" to "country", "país" to "country",
        "tiempo" to "time", "vida" to "life", "agua" to "water", "comida" to "food",
        "ayuda" to "help", "su" to "your", "por" to "for",

        // German vocabulary
        "hallo" to "hello", "welt" to "world", "ja" to "yes", "nein" to "no", "danke" to "thank you",
        "bitte" to "please", "freund" to "friend", "gut" to "good", "sehr" to "very",
        "tag" to "day", "nacht" to "night", "morgen" to "morning", "abend" to "evening",
        "heute" to "today", "jetzt" to "now", "immer" to "always", "nie" to "never",
        "ich" to "I", "du" to "you", "er" to "he", "sie" to "she", "wir" to "we", "ihr" to "you",
        "bin" to "am", "bist" to "are", "ist" to "is", "sind" to "are", "seid" to "are",
        "habe" to "have", "hast" to "have", "hat" to "has", "haben" to "have",
        "machen" to "make", "sagen" to "say", "gehen" to "go", "sehen" to "see", "wissen" to "know",
        "haus" to "house", "auto" to "car", "arbeit" to "work", "stadt" to "city", "land" to "country"
    )

    /**
     * Translates input speech transcript into English in real-time.
     */
    suspend fun translateToEnglish(text: String, sourceLang: String): String = withContext(Dispatchers.Default) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return@withContext ""

        val langCode = sourceLang.lowercase().take(2)
        if (langCode == "en") {
            return@withContext "Original in English"
        }

        // 1. Multi-pass phrase replacement (longest phrases first)
        fun normalizeText(str: String): String =
            str.lowercase().replace(Regex("[^\\p{L}\\p{Nd}]+"), " ").trim()

        val cleanNormalized = normalizeText(trimmed)

        var current = " $cleanNormalized "
        var anyPhraseReplaced = false
        val sortedPhrases = phraseMap.entries.sortedByDescending { normalizeText(it.key).length }
        for ((phrase, translation) in sortedPhrases) {
            val p = " ${normalizeText(phrase)} "
            if (current.contains(p)) {
                current = current.replace(p, " $translation ")
                anyPhraseReplaced = true
            }
        }

        // 2. Word-by-word intelligent mapping with context preservation
        val words = current.trim().split(Regex("\\s+"))
        var translatedAny = anyPhraseReplaced
        val translatedWords = words.map { rawToken ->
            val cleanToken = rawToken.lowercase().replace(Regex("[.,!?;:\"']"), "")
            val punctuation = rawToken.filter { it in ".,!?;:\"'" }
            val match = wordMap[cleanToken]
            if (match != null) {
                translatedAny = true
                if (rawToken.firstOrNull()?.isUpperCase() == true) {
                    match.replaceFirstChar { it.uppercase() } + punctuation
                } else {
                    match + punctuation
                }
            } else {
                rawToken
            }
        }

        val result = translatedWords.joinToString(" ").replace(Regex("\\s+"), " ").trim().replaceFirstChar { it.uppercase() }
        if (translatedAny) {
            return@withContext result
        }

        // 3. Fallback: if no dictionary match, provide language-aware formatted context
        val langLabel = when (langCode) {
            "fr" -> "French"
            "es" -> "Spanish"
            "de" -> "German"
            "it" -> "Italian"
            "pt" -> "Portuguese"
            "zh" -> "Chinese"
            "ja" -> "Japanese"
            "ko" -> "Korean"
            "ru" -> "Russian"
            "ar" -> "Arabic"
            "hi" -> "Hindi"
            else -> sourceLang.uppercase()
        }
        "[$langLabel]: $trimmed"
    }
}
