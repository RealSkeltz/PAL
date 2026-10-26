package nl.pal.client

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.SystemClock
import androidx.core.content.ContextCompat
import com.jiangdg.ausbc.MultiCameraClient
import com.jiangdg.ausbc.callback.IDeviceConnectCallBack
import com.jiangdg.ausbc.callback.IPreviewDataCallBack
import com.jiangdg.ausbc.camera.CameraUVC
import com.jiangdg.ausbc.camera.bean.CameraRequest
import com.jiangdg.usb.USBMonitor
import kotlinx.coroutines.suspendCancellableCoroutine
import livekit.org.webrtc.CapturerObserver
import livekit.org.webrtc.NV21Buffer
import livekit.org.webrtc.SurfaceTextureHelper
import livekit.org.webrtc.VideoCapturer
import livekit.org.webrtc.VideoFrame
import kotlin.coroutines.resume

/**
 * Feeds a USB (UVC) camera into LiveKit. The Pixel doesn't expose USB cameras through
 * Camera2, so AUSBC talks to the device over raw USB and hands us NV21 frames.
 */
class UvcCapturer(private val context: Context, private val device: UsbDevice) : VideoCapturer {

    private var observer: CapturerObserver? = null
    private var client: MultiCameraClient? = null
    private var camera: CameraUVC? = null

    private val frames = object : IPreviewDataCallBack {
        override fun onPreviewData(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {
            if (data == null || format != IPreviewDataCallBack.DataFormat.NV21) return
            val frame = VideoFrame(NV21Buffer(data, width, height, null), 0, SystemClock.elapsedRealtimeNanos())
            observer?.onFrameCaptured(frame)
            frame.release()
        }
    }

    override fun initialize(helper: SurfaceTextureHelper?, context: Context?, observer: CapturerObserver?) {
        this.observer = observer
    }

    override fun startCapture(width: Int, height: Int, fps: Int) {
        val request = CameraRequest.Builder()
            .setPreviewWidth(width)  // AUSBC picks the closest size the camera supports
            .setPreviewHeight(height)
            .setPreviewFormat(CameraRequest.PreviewFormat.FORMAT_MJPEG)
            .setRawPreviewData(true)  // NV21 straight from the camera, no GL readback
            .setAspectRatioShow(false)
            .create()

        val newClient = MultiCameraClient(context, object : IDeviceConnectCallBack {
            override fun onAttachDev(device: UsbDevice?) {}
            override fun onDetachDec(device: UsbDevice?) {
                if (device == this@UvcCapturer.device) PalState.log("USB camera unplugged")
            }
            override fun onConnectDev(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?) {
                if (device != this@UvcCapturer.device || camera != null) return
                camera = CameraUVC(context, this@UvcCapturer.device).apply {
                    setUsbControlBlock(ctrlBlock)
                    addPreviewDataCallBack(frames)
                    openCamera(null, request)  // no view: AUSBC renders off screen
                }
                observer?.onCapturerStarted(true)
            }
            override fun onDisConnectDec(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?) {}
            override fun onCancelDev(device: UsbDevice?) {
                PalState.log("USB camera permission denied")
                observer?.onCapturerStarted(false)
            }
        })
        client = newClient
        newClient.register()
        newClient.requestPermission(device)  // already granted by requestPermission() below, so this just connects
    }

    override fun stopCapture() {
        camera?.removePreviewDataCallBack(frames)
        camera?.closeCamera()
        camera = null
        client?.unRegister()
        client = null
        observer?.onCapturerStopped()
    }

    override fun changeCaptureFormat(width: Int, height: Int, fps: Int) {}

    override fun dispose() {
        stopCapture()
    }

    override fun isScreencast() = false

    companion object {
        private const val ACTION_USB_PERMISSION = "nl.pal.client.USB_PERMISSION"

        /** First attached device with a video interface, or null. */
        fun findCamera(context: Context): UsbDevice? =
            context.getSystemService(UsbManager::class.java).deviceList.values.firstOrNull { device ->
                (0 until device.interfaceCount).any { device.getInterface(it).interfaceClass == UsbConstants.USB_CLASS_VIDEO }
            }

        /**
         * Asks for access to the device, showing the system dialog if needed.
         * Done here rather than through AUSBC, whose immutable PendingIntent on
         * Android 14+ drops the result extras.
         */
        suspend fun requestPermission(context: Context, device: UsbDevice): Boolean {
            val usb = context.getSystemService(UsbManager::class.java)
            if (usb.hasPermission(device)) return true
            return suspendCancellableCoroutine { continuation ->
                val receiver = object : BroadcastReceiver() {
                    override fun onReceive(ctx: Context, intent: Intent) {
                        context.unregisterReceiver(this)
                        if (continuation.isActive) continuation.resume(usb.hasPermission(device))
                    }
                }
                ContextCompat.registerReceiver(
                    context, receiver, IntentFilter(ACTION_USB_PERMISSION), ContextCompat.RECEIVER_NOT_EXPORTED
                )
                continuation.invokeOnCancellation { runCatching { context.unregisterReceiver(receiver) } }
                // Mutable so the system can attach the result; the package makes it explicit, as Android 14+ requires
                val intent = Intent(ACTION_USB_PERMISSION).setPackage(context.packageName)
                val pending = PendingIntent.getBroadcast(context, 0, intent, PendingIntent.FLAG_MUTABLE)
                usb.requestPermission(device, pending)
            }
        }
    }
}
