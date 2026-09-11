package com.edgeort.android.ui

import android.annotation.SuppressLint
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.view.*
import android.widget.TextView
import androidx.cardview.widget.CardView
import com.edgeort.android.SpeechRecognitionService
import com.edgeort.android.SpeechRecognitionService.PipelineUpdate
import kotlinx.coroutines.*

/**
 * System-Wide Floating Live Caption & Subtitle Overlay.
 * Floats a draggable translucent card over third-party applications (YouTube, Zoom, Calls)
 * displaying real-time speech transcription and English translations.
 */
class FloatingSubtitleOverlayService : Service() {

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager

        val layoutType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            layoutType,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            y = 150 // Distance from bottom
        }

        // Programmatically construct the translucent floating subtitle card
        val card = CardView(this).apply {
            radius = 24f
            cardElevation = 16f
            setCardBackgroundColor(0xD91E1E1E.toInt()) // 85% opacity dark grey
        }

        val textView = TextView(this).apply {
            setPadding(32, 24, 32, 24)
            textSize = 17f
            setTextColor(0xFFFFFFFF.toInt())
            text = "Edge ORT Subtitles: Listening…"
        }

        card.addView(textView)
        overlayView = card

        // Draggable touch handling
        card.setOnTouchListener { view, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = params.x
                    initialY = params.y
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    params.x = initialX + (event.rawX - initialTouchX).toInt()
                    params.y = initialY - (event.rawY - initialTouchY).toInt()
                    windowManager?.updateViewLayout(view, params)
                    true
                }
                else -> false
            }
        }

        windowManager?.addView(overlayView, params)

        // Observe real-time speech events
        serviceScope.launch {
            SpeechRecognitionService.events.collect { update ->
                when (update) {
                    is PipelineUpdate.Transcript -> {
                        val text = if (!update.translation.isNullOrBlank()) {
                            "${update.text}\n↳ ${update.translation}"
                        } else {
                            update.text
                        }
                        textView.text = text
                    }
                    is PipelineUpdate.Status -> {
                        if (textView.text.isEmpty()) {
                            textView.text = update.message
                        }
                    }
                    is PipelineUpdate.Error -> {
                        textView.text = "Error: ${update.message}"
                    }
                    else -> {}
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        overlayView?.let {
            windowManager?.removeView(it)
            overlayView = null
        }
    }
}
