// ============================================================
// BTC movement "predictor" for ESP32-C3 + SSD1306 OLED
// ------------------------------------------------------------
// Ports the desktop n-gram experiment onto the microcontroller:
//   - fetches the live BTC price over WiFi (CoinGecko, HTTPS)
//   - discretizes price change into 5 movement tokens, same
//     thresholds as the Python version
//   - learns a first-order Markov transition matrix between
//     tokens (this replaces the desktop trigram+vector model —
//     a 5x5 table is the embedded-appropriate equivalent)
//   - predicts the next movement, then verifies it against the
//     next real price fetch and tracks rolling accuracy
//   - shows price / prediction / accuracy on the OLED
//
// DISCLAIMER: same as the desktop version. This has no real
// predictive edge on BTC price. It's an embedded systems + toy
// statistics demo, not a trading tool.
//
// Hardware assumed: ESP32-C3 board with an SSD1306 OLED wired
// as in your original code (I2C, SDA=5, SCL=6). This matches
// the Seeed XIAO ESP32-C3 + OLED expansion board form factor
// (128x64 driver buffer, ~72x40 visible window) — adjust
// width/height/offsets below if your panel differs.
//
// Libraries needed (Arduino Library Manager):
//   - U8g2 (oliver)
//   - ArduinoJson (Benoit Blanchon), v6.x
//   (WiFi.h, HTTPClient.h, WiFiClientSecure.h, Preferences.h
//    ship with the ESP32 Arduino core)
// ============================================================

#include <U8g2lib.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Preferences.h>

// WiFiClientSecure's TLS handshake needs more stack than the ESP32
// Arduino core gives the loop task by default (8KB). Undersized stack
// here is a common cause of a silent crash/reboot loop — which looks
// exactly like "nothing shows up on the OLED," because the crash
// happens before the first draw call. Must be called before setup().
SET_LOOP_TASK_STACK_SIZE(16 * 1024);

// ---------------- WiFi ----------------
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";

// ---------------- OLED (unchanged from your setup) ----------------
#define OLED_RESET U8X8_PIN_NONE
#define OLED_SDA 5
#define OLED_SCL 6
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, OLED_RESET, OLED_SCL, OLED_SDA);

const int width = 72;
const int height = 40;
const int xOffset = 30;  // = (132-w)/2
const int yOffset = 12;  // = (64-h)/2

// ---------------- Timing ----------------
// CoinGecko's free tier is fine with this; don't go much faster.
const unsigned long FETCH_INTERVAL_MS = 5UL * 60UL * 1000UL;  // 5 minutes
unsigned long lastFetchMs = 0;

// ---------------- Movement tokens ----------------
enum Token : int8_t {
  STRONG_DOWN = 0,
  DOWN        = 1,
  FLAT        = 2,
  UP          = 3,
  STRONG_UP   = 4,
  NUM_TOKENS  = 5,
  TOKEN_NONE  = -1
};

const char* TOKEN_LABEL[NUM_TOKENS] = {"STR.DN", "DOWN", "FLAT", "UP", "STR.UP"};
// Midpoint % used to turn a predicted token back into a price guess.
const float TOKEN_MIDPOINT_PCT[NUM_TOKENS] = {-2.5f, -0.8f, 0.0f, 0.8f, 2.5f};

int8_t classifyMovement(float pctChange) {
  if (pctChange < -1.5f) return STRONG_DOWN;
  if (pctChange < -0.3f) return DOWN;
  if (pctChange <  0.3f) return FLAT;
  if (pctChange <  1.5f) return UP;
  return STRONG_UP;
}

// ---------------- Persisted state ----------------
Preferences prefs;

uint16_t transition[NUM_TOKENS][NUM_TOKENS];  // transition[from][to] counts
int8_t   lastToken     = TOKEN_NONE;
float    lastPrice     = -1.0f;
int8_t   predictedToken = TOKEN_NONE;
float    predictedPrice = 0.0f;
uint32_t correctCount  = 0;
uint32_t totalCount    = 0;
uint32_t uptimeSeconds = 0;

void loadState() {
  prefs.begin("btcpred", false);
  prefs.getBytes("transition", transition, sizeof(transition));
  lastToken      = prefs.getChar("lastToken", TOKEN_NONE);
  lastPrice      = prefs.getFloat("lastPrice", -1.0f);
  predictedToken = prefs.getChar("predToken", TOKEN_NONE);
  predictedPrice = prefs.getFloat("predPrice", 0.0f);
  correctCount   = prefs.getUInt("correct", 0);
  totalCount     = prefs.getUInt("total", 0);
  prefs.end();
}

void saveState() {
  prefs.begin("btcpred", false);
  prefs.putBytes("transition", transition, sizeof(transition));
  prefs.putChar("lastToken", lastToken);
  prefs.putFloat("lastPrice", lastPrice);
  prefs.putChar("predToken", predictedToken);
  prefs.putFloat("predPrice", predictedPrice);
  prefs.putUInt("correct", correctCount);
  prefs.putUInt("total", totalCount);
  prefs.end();
}

// ---------------- Prediction ----------------
int8_t pickNextToken(int8_t fromToken) {
  if (fromToken == TOKEN_NONE) {
    return (Token)random(0, NUM_TOKENS);  // no history yet: uniform guess
  }
  uint32_t rowTotal = 0;
  for (int t = 0; t < NUM_TOKENS; t++) rowTotal += transition[fromToken][t];
  if (rowTotal == 0) {
    return (Token)random(0, NUM_TOKENS);  // never seen this state before
  }
  long r = random(0, rowTotal);
  uint32_t acc = 0;
  for (int t = 0; t < NUM_TOKENS; t++) {
    acc += transition[fromToken][t];
    if (r < (long)acc) return (Token)t;
  }
  return (Token)(NUM_TOKENS - 1);
}

// ---------------- Networking ----------------
void connectWiFi() {
  Serial.println("[wifi] connecting...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[wifi] connected, IP=");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[wifi] FAILED to connect within 20s");
  }
}

// Returns true and sets outPrice on success.
bool fetchBtcPrice(float& outPrice) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) return false;
  }

  WiFiClientSecure client;
  client.setInsecure();  // skip cert validation — simplest option for a
                          // hobby device; use a pinned root CA if this
                          // matters for your use case.

  HTTPClient https;
  https.setConnectTimeout(8000);
  https.setTimeout(8000);
  const char* url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd";
  Serial.println("[http] begin...");
  if (!https.begin(client, url)) {
    Serial.println("[http] begin() failed");
    return false;
  }

  Serial.println("[http] GET...");
  int code = https.GET();
  Serial.print("[http] status code: ");
  Serial.println(code);
  if (code != HTTP_CODE_OK) {
    https.end();
    return false;
  }

  String payload = https.getString();
  https.end();
  Serial.print("[http] payload: ");
  Serial.println(payload);

  JsonDocument doc;  // ArduinoJson v7: unified, auto-sized document
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.print("[json] parse failed: ");
    Serial.println(err.c_str());
    return false;
  }

  outPrice = doc["bitcoin"]["usd"].as<float>();
  Serial.print("[price] ");
  Serial.println(outPrice);
  return outPrice > 0.0f;
}

// ---------------- One fetch/predict/verify cycle ----------------
void runCycle() {
  float currentPrice;
  if (!fetchBtcPrice(currentPrice)) return;  // skip this cycle silently, try again next interval

  if (lastPrice > 0.0f) {
    float pctChange = (currentPrice - lastPrice) / lastPrice * 100.0f;
    int8_t actualToken = classifyMovement(pctChange);

    // Verify the prediction made last cycle, if any.
    if (predictedToken != TOKEN_NONE) {
      totalCount++;
      float predictedDir = predictedPrice - lastPrice;
      float actualDir = currentPrice - lastPrice;
      bool directionCorrect = (predictedDir * actualDir) >= 0.0f;
      if (directionCorrect) correctCount++;
    }

    // Learn: update transition counts from the previous state to this one.
    if (lastToken != TOKEN_NONE) {
      transition[lastToken][actualToken]++;
    }
    lastToken = actualToken;
  }

  lastPrice = currentPrice;

  // Make the next prediction from the current state.
  int8_t next = pickNextToken(lastToken);
  predictedToken = next;
  predictedPrice = currentPrice * (1.0f + TOKEN_MIDPOINT_PCT[next] / 100.0f);

  saveState();
}

// ---------------- Display ----------------
// Single-line status splash — used during boot/connect so the screen
// is never sitting blank while WiFi or HTTP calls are in progress.
void showStatus(const char* line1, const char* line2 = "") {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_4x6_tr);
  u8g2.drawStr(xOffset, yOffset + 14, line1);
  if (line2 && line2[0]) {
    u8g2.drawStr(xOffset, yOffset + 22, line2);
  }
  u8g2.sendBuffer();
}

void updateDisplay() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_4x6_tr);

  char line[24];

  bool wifiOk = WiFi.status() == WL_CONNECTED;
  snprintf(line, sizeof(line), "WiFi: %s", wifiOk ? "OK" : "...");
  u8g2.drawStr(xOffset, yOffset + 6, line);

  if (lastPrice > 0.0f) {
    snprintf(line, sizeof(line), "BTC $%d", (int)lastPrice);
  } else {
    snprintf(line, sizeof(line), "BTC: --");
  }
  u8g2.drawStr(xOffset, yOffset + 14, line);

  if (predictedToken != TOKEN_NONE) {
    snprintf(line, sizeof(line), "Next: %s", TOKEN_LABEL[predictedToken]);
  } else {
    snprintf(line, sizeof(line), "Next: --");
  }
  u8g2.drawStr(xOffset, yOffset + 22, line);

  if (totalCount > 0) {
    int acc = (int)(100.0f * correctCount / totalCount);
    snprintf(line, sizeof(line), "Acc:%d%% n=%lu", acc, (unsigned long)totalCount);
  } else {
    snprintf(line, sizeof(line), "Acc: n/a");
  }
  u8g2.drawStr(xOffset, yOffset + 30, line);

  snprintf(line, sizeof(line), "Up:%lus", (unsigned long)uptimeSeconds);
  u8g2.drawStr(xOffset, yOffset + 38, line);

  u8g2.sendBuffer();
}

// ---------------- Setup / loop ----------------
void setup(void) {
  Serial.begin(115200);
  delay(300);  // let USB serial settle
  Serial.println("\n[boot] starting");

  u8g2.begin();
  u8g2.setContrast(255);
  u8g2.setBusClock(400000);
  showStatus("Booting...");  // prove the display works before anything risky runs

  randomSeed(esp_random());
  loadState();

  showStatus("Connecting", "to WiFi...");
  connectWiFi();

  if (WiFi.status() == WL_CONNECTED) {
    showStatus("Fetching", "BTC price...");
  } else {
    showStatus("WiFi FAILED", "check creds");
  }

  // Run one cycle immediately so the display has something to show
  // right away instead of waiting a full interval.
  runCycle();
  lastFetchMs = millis();
  Serial.println("[boot] setup complete");
}

void loop(void) {
  unsigned long now = millis();
  if (now - lastFetchMs >= FETCH_INTERVAL_MS) {
    runCycle();
    lastFetchMs = now;
  }
  uptimeSeconds = millis() / 1000;
  updateDisplay();
  delay(1000);
}
