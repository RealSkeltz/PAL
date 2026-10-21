package nl.pal.client

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {

    private val prefs by lazy { getSharedPreferences("Pal", MODE_PRIVATE) }

    private val requiredPermissions = buildList {
        add(Manifest.permission.RECORD_AUDIO)
        add(Manifest.permission.CAMERA)
        add(Manifest.permission.BLUETOOTH_CONNECT)
        if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
        if (Build.VERSION.SDK_INT >= 37) add(Manifest.permission.ACCESS_LOCAL_NETWORK)
    }.toTypedArray()

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            val ok = result[Manifest.permission.RECORD_AUDIO] == true &&
                result[Manifest.permission.CAMERA] == true &&
                (Build.VERSION.SDK_INT < 37 || result[Manifest.permission.ACCESS_LOCAL_NETWORK] == true)
            if (ok) startPal() else PalState.log("Mic, camera and local network permission are required")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                PalScreen(
                    initialHost = prefs.getString("host", "") ?: "",
                    onConnect = { host ->
                        prefs.edit().putString("host", host.trim()).apply()
                        permissionRequest.launch(requiredPermissions)
                    },
                    onDisconnect = { stopService(Intent(this, PalService::class.java)) },
                    onPing = {
                        startService(Intent(this, PalService::class.java).setAction(PalService.ACTION_PING))
                    },
                )
            }
        }
    }

    private fun startPal() {
        val intent = Intent(this, PalService::class.java)
            .putExtra(PalService.EXTRA_HOST, prefs.getString("host", ""))
        ContextCompat.startForegroundService(this, intent)
    }
}

@Composable
fun PalScreen(
    initialHost: String,
    onConnect: (String) -> Unit,
    onDisconnect: () -> Unit,
    onPing: () -> Unit,
) {
    val status by PalState.status.collectAsState()
    val lines by PalState.lines.collectAsState()
    var host by remember { mutableStateOf(initialHost) }
    val running = status != "Offline" && status != "Couldn't connect" && status != "Disconnected"

    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier
                .safeDrawingPadding()
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Pal", fontSize = 28.sp, fontWeight = FontWeight.Bold)
            Text(
                status,
                fontSize = 36.sp,
                fontWeight = FontWeight.Bold,
                color = if (status == "Live") MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurface,
            )

            OutlinedTextField(
                value = host,
                onValueChange = { host = it },
                label = { Text("Server Tailscale IP (e.g. 100.64.0.2)") },
                singleLine = true,
                enabled = !running,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                modifier = Modifier.fillMaxWidth(),
            )

            if (!running) {
                Button(onClick = { onConnect(host) }, enabled = host.isNotBlank(), modifier = Modifier.fillMaxWidth()) {
                    Text("Connect")
                }
            } else {
                Button(onClick = onPing, modifier = Modifier.fillMaxWidth()) { Text("Ping PAL") }
                OutlinedButton(onClick = onDisconnect, modifier = Modifier.fillMaxWidth()) { Text("Disconnect") }
            }

            LazyColumn(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                items(lines) { line ->
                    Text(line, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}
