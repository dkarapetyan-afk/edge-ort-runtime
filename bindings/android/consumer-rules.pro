# Keep UniFFI generated classes and native methods
-keep class uniffi.** { *; }
-keep class com.edgeort.android.** { *; }
-keepclassmembers class * {
    native <methods>;
}
