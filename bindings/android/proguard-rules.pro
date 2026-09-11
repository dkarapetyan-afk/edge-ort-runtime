# ProGuard rules for edge-ort-android library
-keep class uniffi.** { *; }
-keep class com.edgeort.android.** { *; }
-keepclassmembers class * {
    native <methods>;
}
