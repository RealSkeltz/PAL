package nl.pal.client

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.media.ToneGenerator
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import io.livekit.android.LiveKit
import io.livekit.android.events.RoomEvent
import io.livekit.android.events.collect
import io.livekit.android.room.Room
import io.livekit.android.room.track.CameraPosition
import io.livekit.android.room.track.DataPublishReliability
import io.livekit.android.room.track.LocalVideoTrackOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Foreground service that keeps mic + camera streaming to the Pal server,
 * also with the screen off or the phone in a pocket.
 *
 * Audio routing: LiveKit automatically prefers a connected Bluetooth headset,
 * so pair the earpiece before connecting.
 */
class PalService : LifecycleService() {

    private var room: Room? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)

        if (intent?.action == ACTION_PING) {
            if (room == null) stopSelf()  // not connected: don't leave a service running outside the foreground
            else sendEvent("""{"type":"ping","t":${System.currentTimeMillis()}}""")
            return START_NOT_STICKY
        }
        if (room != null) return START_NOT_STICKY  // already running

        val host = intent?.getStringExtra(EXTRA_HOST)
        if (host.isNullOrBlank()) {
            stopSelf()
            return START_NOT_STICKY
        }

        goForeground()
        connect(host)
        return START_NOT_STICKY
    }

    private fun goForeground() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Pal", NotificationManager.IMPORTANCE_LOW)
        )
        val openApp = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Pal is active")
            .setContentText("Streaming mic and camera")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .setContentIntent(openApp)
            .build()

        startForeground(
            NOTIFICATION_ID, notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE or ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
        )
    }

    private fun connect(host: String) = lifecycleScope.launch {
        try {
            PalState.status.value = "Connecting"
            val token = fetchToken(host)  // served by `python main.py --remote`

            val newRoom = LiveKit.create(applicationContext)
            room = newRoom
            newRoom.videoTrackCaptureDefaults = LocalVideoTrackOptions(position = CameraPosition.BACK)

            // Events from the server and the connection
            launch {
                newRoom.events.collect { event ->
                    when (event) {
                        is RoomEvent.TrackSubscribed -> PalState.log("Receiving Pal's voice")
                        is RoomEvent.DataReceived -> handleServerEvent(String(event.data))
                        is RoomEvent.Reconnecting -> {
                            PalState.status.value = "Reconnecting"
                            playCue(ToneGenerator.TONE_PROP_NACK)  // generated on the phone
                        }
                        is RoomEvent.Reconnected -> {
                            PalState.status.value = "Live"
                            playCue(ToneGenerator.TONE_PROP_ACK)
                        }
                        is RoomEvent.Disconnected -> {
                            PalState.status.value = "Disconnected"
                            stopSelf()
                        }
                        else -> {}
                    }
                }
            }

            newRoom.connect("ws://$host:7880", token)
            newRoom.localParticipant.setMicrophoneEnabled(true)
            publishCamera(newRoom)

            PalState.status.value = "Live"
            PalState.log("Connected to $host")
        } catch (e: Exception) {
            PalState.status.value = "Couldn't connect"
            PalState.log("Error: ${e.message}")
            stopSelf()
        }
    }

    /** The USB chest camera when one is plugged in, otherwise the phone's back camera. */
    private suspend fun publishCamera(room: Room) {
        val device = UvcCapturer.findCamera(this)
        if (device == null || !UvcCapturer.requestPermission(this, device)) {
            room.localParticipant.setCameraEnabled(true)
            PalState.log("Using the phone camera")
            return
        }
        val track = room.localParticipant.createVideoTrack("usb-camera", UvcCapturer(applicationContext, device))
        track.startCapture()
        room.localParticipant.publishVideoTrack(track)
        PalState.log("Using the USB camera")
    }

    private fun handleServerEvent(json: String) {
        // PAL's events: {"type": "pong"} for now; later state changes, transcripts, tool results...
        PalState.log("Server: $json")
    }

    private fun sendEvent(json: String) {
        val currentRoom = room ?: return
        lifecycleScope.launch {
            currentRoom.localParticipant.publishData(
                json.toByteArray(), DataPublishReliability.RELIABLE, topic = "events"
            )
            PalState.log("Sent ping")
        }
    }

    private suspend fun fetchToken(host: String): String = withContext(Dispatchers.IO) {
        val connection = URL("http://$host:8000/token?identity=pixel").openConnection() as HttpURLConnection
        connection.connectTimeout = 5000
        connection.readTimeout = 5000
        connection.inputStream.bufferedReader().use { JSONObject(it.readText()).getString("token") }
    }

    private fun playCue(tone: Int) {
        val generator = ToneGenerator(AudioManager.STREAM_VOICE_CALL, 80)
        generator.startTone(tone, 300)
        lifecycleScope.launch {
            delay(400)
            generator.release()
        }
    }

    override fun onDestroy() {
        room?.disconnect()
        room?.release()
        room = null
        PalState.status.value = "Offline"
        super.onDestroy()
    }

    companion object {
        const val EXTRA_HOST = "host"
        const val ACTION_PING = "nl.pal.client.PING"
        private const val CHANNEL_ID = "Pal"
        private const val NOTIFICATION_ID = 1
    }
}
