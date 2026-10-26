# PAL Android client: setup

The app is a thin client. It streams the earpiece mic and the phone's back camera to
PAL over LiveKit, plays PAL's voice back, and exchanges events on the data channel.
It runs as a foreground service, so it keeps working with the screen off.

```
phone app ──mic, camera──► livekit-server ──► python main.py --remote
          ◄──PAL's voice──                ◄──
          ──GET :8000/token────────────────►  (served by main.py too)
```

All the server-side logic lives in `pal/remote.py`; this folder only holds the client.

## 1. Server (laptop)

Shortcut: `./start-remote.sh` from the repo root does all of the below in one go. It
uses the Tailscale IP when Tailscale is installed and the Wi-Fi IP otherwise, starts
Ollama if needed, prints the address to type into the app, and stops everything on Ctrl+C.
Extra arguments go to `main.py`, e.g. `./start-remote.sh --verbose`. By hand:

```bash
brew install livekit
livekit-server --dev --bind 0.0.0.0 --node-ip "$(tailscale ip -4)"

# second terminal, repo root
poetry run python main.py --remote --preview
```

`--remote` replaces the laptop mic, webcam and speaker with the phone's; everything
else (STT, speaker ID, routing, vision, TTS) runs as usual. `--preview` shows the
phone's camera in the browser. The dev keys (`devkey` / `secret`) are the defaults;
set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` to use another server.

No `tailscale serve` or HTTPS needed: Tailscale already encrypts the traffic.
The token endpoint listens on all interfaces, so anyone on the same network can join
the room. Fine on a home network or tailnet; don't run it on public Wi-Fi.

## 2. Phone prep

1. Install the **Tailscale** app on the Pixel and log in with the same account as the laptop.
1. Enable **Developer options**: Settings → About phone → tap _Build number_ 7 times.
1. Settings → System → Developer options → turn on **USB debugging**.
1. Pair the Bluetooth earpiece.

Quick check: with `main.py --remote` running, open `http://<laptop-tailscale-ip>:8000/token`
in the phone's browser. You should see a JSON token. If not, the network path is the
problem, not the app.

## 3. Create the project in Android Studio

```bash
brew install --cask android-studio
```

1. Open Android Studio, let it download the SDK, then **New Project → Empty Activity**.
1. Name `Pal`, package name `nl.pal.client`, language Kotlin,
   minimum SDK **API 31**, build configuration language Kotlin DSL.
   The package name must match the `package` line in the `.kt` files, and the
   project name `Pal` is what creates the theme `Theme.Pal` the manifest expects.
1. Let the first Gradle sync finish.

## 4. Add LiveKit

In `settings.gradle.kts`, inside `dependencyResolutionManagement { repositories { ... } }`, add:

```kotlin
maven { url = uri("https://jitpack.io") }
```

In `app/build.gradle.kts`, inside `dependencies { ... }`, add:

```kotlin
implementation("io.livekit:livekit-android:2.+")   // pin to the version Gradle resolves once it works
implementation("androidx.lifecycle:lifecycle-service:2.8.7")
// USB chest camera: the Pixel doesn't expose USB cameras through Camera2, so AUSBC
// reads them over raw USB. libuvc is listed too because its types appear in AUSBC's API.
implementation("com.github.ernestp.AndroidUSBCamera:libausbc:3.6.0")
implementation("com.github.ernestp.AndroidUSBCamera:libuvc:3.6.0")
```

Also check that `minSdk = 31` in the same file. Sync Gradle.

## 5. Drop in the files

Copy from `client/` into the project's `app/src/main/`:

- `AndroidManifest.xml` (replaces the generated one)
- `MainActivity.kt` → `java/nl/pal/client/MainActivity.kt` (replaces the generated one)
- `PalService.kt` → `java/nl/pal/client/PalService.kt`
- `PalState.kt` → `java/nl/pal/client/PalState.kt`
- `UvcCapturer.kt` → `java/nl/pal/client/UvcCapturer.kt`

The template's `ui/theme` folder can stay; it's just unused.

## 6. Put it on the phone

1. Connect the Pixel by USB and accept the "Allow USB debugging?" prompt on the phone.
1. Select the Pixel in Android Studio's device dropdown and press **Run ▶**.
   The app installs and opens. It stays on the phone afterwards like any other app.
1. Wireless alternative: Developer options → _Wireless debugging_, then
   _Pair devices using Wi-Fi_ in Android Studio's device manager.

## 7. Use it

1. Enter the laptop's Tailscale IP, tap **Connect**, grant mic, camera,
   Bluetooth and notification permissions.
1. You should hear a chime in the earpiece, and PAL's log shows `device joined`,
   `hearing ...'s mic` and `seeing ...'s camera`.
1. With the USB camera plugged in, the app asks for access to it and logs
   `Using the USB camera`. Without one it falls back to the phone's back camera.
1. Talk to PAL as usual; it answers in the earpiece.
1. **Ping PAL** should come back as `{"type":"pong",...}` in the app's log.
1. Lock the screen: the notification stays and streaming continues.

## Troubleshooting

- **Connects but no audio either way:** usually UDP. Check that `--node-ip` is the
  laptop's Tailscale IP, not its Wi-Fi IP.
- **Chime plays on the phone speaker:** connect the earpiece _before_ tapping Connect.
- **"failed to connect ... after 5000ms" while the browser check works:** Android 17
  blocks apps from the home network without the local network permission. Grant it
  when the app asks, or in Settings → Apps → Pal → Permissions.
- **"Couldn't connect" immediately:** the token request failed; redo the browser check
  in step 2.
- **PAL hears you but never treats it as you:** the voiceprint was enrolled on the
  laptop mic, and a Bluetooth earpiece sounds different. Re-enroll through the earpiece
  if speaker scores stay low (they're in PAL's log).
- **App sometimes crashes when disconnecting with the USB camera on:** known bug in
  AUSBC's bundled libuvc (`pthread_mutex_lock called on a destroyed mutex`): stopping the
  stream frees it while a USB transfer is still completing. Streaming itself isn't affected;
  reopen the app. Upstream: https://github.com/ernestp/AndroidUSBCamera/issues/10
- **USB camera feed slows to a few fps:** cheap UVC cameras lower their frame rate in dim
  light, and loose cables or low power make the camera drop off USB. Try better light or a
  powered USB-C hub.
- **Build errors on LiveKit classes:** SDK APIs shift slightly between versions.
  Check the LiveKit Android SDK docs for the version Gradle resolved.
