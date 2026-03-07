"""
Telegram bot for Kalshi bet notifications and approvals.
Sends opportunities, receives confirm/skip responses.
"""

import logging
import os
import threading

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set. "
                "Create a bot via @BotFather and get your chat ID via @userinfobot"
            )
        self.base_url = TELEGRAM_API.format(token=self.token)
        self._last_update_id = 0
        self._callbacks = {}  # callback_id -> callback_fn
        self._poll_thread = None
        self._stop_event = threading.Event()

    def send_message(self, text, parse_mode="HTML", reply_markup=None):
        """Send a message to the configured chat."""
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return None

    def send_opportunity(self, opp, callback_id):
        """Send a bet opportunity with Place/Skip inline buttons."""
        conf = opp.get("confidence", 0)
        conf_emoji = "🔥" if conf >= 70 else ("📊" if conf >= 60 else "📋")
        profit = opp.get("potential_profit_cents", 0)
        text = (
            f"{conf_emoji} <b>Kalshi — {opp['bet_type'].upper()}</b>\n\n"
            f"<b>{opp['away_team']} @ {opp['home_team']}</b>\n"
            f"Pick: <b>{opp['pick']}</b>\n"
            f"Confidence: <b>{conf}%</b>\n"
            f"Kalshi: {opp['kalshi_price']}c\n"
            f"Cost: {opp['contracts']}x @ {opp['price_cents']}c "
            f"(${opp['stake_cents'] / 100:.2f})\n"
            f"Profit if win: <b>${profit / 100:.2f}</b>\n"
            f"Volume: {opp.get('volume', 0):,}"
        )

        reply_markup = {
            "inline_keyboard": [[
                {"text": "Place Bet", "callback_data": f"place:{callback_id}"},
                {"text": "Skip", "callback_data": f"skip:{callback_id}"},
            ]]
        }

        return self.send_message(text, reply_markup=reply_markup)

    def send_bet_result(self, opp, success, message=""):
        """Send confirmation that a bet was placed or skipped."""
        if success:
            text = (
                f"<b>Bet Placed</b>\n"
                f"{opp['pick']} ({opp['bet_type'].upper()})\n"
                f"{opp['contracts']}x @ {opp['price_cents']}c\n"
                f"{message}"
            )
        else:
            text = f"<b>Bet Failed</b>\n{opp.get('pick', 'Unknown')}\n{message}"
        self.send_message(text)

    def send_daily_summary(self, opportunities, balance_cents=None):
        """Send a daily summary of opportunities found."""
        if not opportunities:
            text = (
                "<b>Daily Kalshi Scan</b>\n\n"
                "No opportunities with 10%+ edge found today."
            )
            if balance_cents is not None:
                text += f"\nBalance: ${balance_cents / 100:.2f}"
            self.send_message(text)
            return

        text = f"<b>Daily Kalshi Scan</b>\n\n"
        if balance_cents is not None:
            text += f"Balance: ${balance_cents / 100:.2f}\n"
        text += f"Found <b>{len(opportunities)}</b> opportunities:\n\n"

        for i, opp in enumerate(opportunities, 1):
            text += (
                f"{i}. {opp['pick']} ({opp['bet_type'].upper()}) "
                f"{opp.get('confidence', 0)}% conf | "
                f"{opp['contracts']}x @ {opp['price_cents']}c\n"
            )

        text += "\nBet notifications will follow with Place/Skip buttons."
        self.send_message(text)

    def register_callback(self, callback_id, fn):
        """Register a callback function for a button press."""
        self._callbacks[callback_id] = fn

    def _process_updates(self):
        """Poll for callback query updates (button presses)."""
        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={
                    "offset": self._last_update_id + 1,
                    "timeout": 30,
                    "allowed_updates": '["callback_query"]',
                },
                timeout=35,
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        except Exception as e:
            logger.warning(f"Telegram poll error: {e}")
            return

        for update in updates:
            self._last_update_id = update["update_id"]
            callback_query = update.get("callback_query")
            if not callback_query:
                continue

            data = callback_query.get("data", "")
            query_id = callback_query.get("id")

            parts = data.split(":", 1)
            if len(parts) != 2:
                continue
            action, callback_id = parts

            # Verify it's from the right chat
            msg_chat = callback_query.get("message", {}).get("chat", {}).get("id")
            if str(msg_chat) != str(self.chat_id):
                continue

            # Answer the callback query (removes loading indicator)
            try:
                requests.post(
                    f"{self.base_url}/answerCallbackQuery",
                    json={"callback_query_id": query_id, "text": "Processing..."},
                    timeout=5,
                )
            except Exception:
                pass

            fn = self._callbacks.pop(callback_id, None)
            if fn:
                try:
                    fn(action)
                except Exception as e:
                    logger.error(f"Telegram callback error: {e}")

    def start_polling(self):
        """Start background thread to listen for button presses."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._stop_event.clear()

        def poll_loop():
            while not self._stop_event.is_set():
                self._process_updates()
                if self._stop_event.wait(1):
                    break

        self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("Telegram bot polling started")

    def stop_polling(self):
        self._stop_event.set()
