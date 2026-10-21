package nl.pal.client

import kotlinx.coroutines.flow.MutableStateFlow

/** Shared state between the service (does the work) and the activity (shows it). */
object PalState {
    val status = MutableStateFlow("Offline")
    val lines = MutableStateFlow<List<String>>(emptyList())

    fun log(message: String) {
        lines.value = (listOf(message) + lines.value).take(50)
    }
}
