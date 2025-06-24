import asyncio
import json
import os
import datetime
import uuid
from typing import Dict, List, Optional
from functools import lru_cache
import logging
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

logging.basicConfig(
    filename="tcm_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    print("警告：未找到 OPENWEATHER_API_KEY，天氣相關功能將無法使用。")
    # logging.warning("未找到 OPENWEATHER_API_KEY，天氣相關功能將無法使用。")

# === 中醫舌診專業 System Prompt (TCM Tongue Diagnosis System Prompt) ===
TCM_SYSTEM_PROMPT = """
你是一位專業的中醫舌診智能助手，具備以下專業知識和特點：

🏥 【專業背景】
- 精通中醫舌診理論，熟悉《中醫診斷學》舌診篇章
- 了解舌質、舌苔、舌態的診斷意義
- 掌握舌象與體質、病證的對應關係
- 具備豐富的中醫食療和養生調理知識

🎯 【服務宗旨】
- 提供專業、準確的舌象分析和體質評估
- 給出個性化的中醫養生建議和食療方案
- 追蹤用戶健康變化，提供趨勢分析
- 結合現代生活環境（天氣、季節）給出調理建議

📝 【回應原則】
- 所有回應必須使用繁體中文
- 語言親切專業，避免過於艱深的醫學術語
- 提供具體可行的建議，避免空泛的建議
- 強調中醫養生重在調理，非急症治療
- 適時提醒用戶如有嚴重症狀應就醫

⚠️【重要聲明】
- 舌診分析僅供參考，不可替代專業醫療診斷
- 嚴重或持續的健康問題應諮詢專業中醫師
- 體質調理需要時間，建議長期堅持

🌿 【中醫理念】
- 重視「治未病」的預防醫學理念
- 強調「辨證論治」的個體化調理
- 注重「天人合一」的整體觀念
- 提倡「藥食同源」的養生方法

請以溫和、專業、關懷的語調與用戶互動，並始終使用繁體中文回應。

⚠️ 當用戶輸入任何舌象描述（如「舌頭有紅點」、「舌苔黃厚」等），請務必呼叫舌象分析工具（enhanced_tongue_analysis），不要自行生成分析內容。
"""

llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
)

TONGUE_DATABASE = {
    "舌頭有紅點": {
        "體質": "內熱過盛",
        "症狀": ["口乾", "煩躁", "失眠"],
        "建議": "清熱降火，避免辛辣食物",
        "食療": ["菊花茶", "蓮子心茶", "綠豆湯"],
        "禁忌": ["辛辣", "油炸", "燒烤"],
        "中醫理論": "舌起紅點多為心火上炎或胃熱熾盛之象"
    },
    "舌尖及側邊發紅": {
        "體質": "肝氣鬱結",
        "症狀": ["情緒不穩", "脅痛", "月經不調"],
        "建議": "疏肝理氣，保持心情愉快",
        "食療": ["玫瑰花茶", "陳皮茶", "柴胡疏肝散"],
        "禁忌": ["過度勞累", "情緒激動"],
        "中醫理論": "舌邊尖紅為肝膽火旺，情志不遂所致"
    },
    "舌頭有齒痕": {
        "體質": "脾虛濕重",
        "症狀": ["疲勞", "腹脹", "大便溏薄"],
        "建議": "健脾祛濕，適量運動",
        "食療": ["四神湯", "薏仁水", "茯苓餅"],
        "禁忌": ["生冷食物", "甜膩食物"],
        "中醫理論": "舌邊有齒痕乃脾氣虛弱，舌體胖嫩所致"
    },
    "舌頭有溝痕": {
        "體質": "脾胃虛弱",
        "症狀": ["食慾不振", "消化不良", "面色萎黃"],
        "建議": "健脾養胃，規律飲食",
        "食療": ["山藥粥", "蓮子湯", "白朮茶"],
        "禁忌": ["過飽過餓", "冰冷飲品"],
        "中醫理論": "舌有裂紋多為陰虛或脾胃氣虛，津液不足"
    },
    "舌苔黃厚": {
        "體質": "濕熱體質",
        "症狀": ["口苦", "小便黃", "大便黏膩"],
        "建議": "清熱利濕，飲食清淡",
        "食療": ["茵陳蒿茶", "綠豆薏仁湯", "冬瓜湯"],
        "禁忌": ["油膩食物", "酒類", "甜食"],
        "中醫理論": "苔黃而厚為胃腸積熱，濕熱內蘊之象"
    },
    "舌苔白厚": {
        "體質": "寒濕體質",
        "症狀": ["畏寒", "腹脹", "大便溏薄"],
        "建議": "溫陽化濕，避免生冷",
        "食療": ["生薑茶", "肉桂茶", "附子理中湯"],
        "禁忌": ["生冷食物", "寒涼水果"],
        "中醫理論": "苔白而厚為寒濕內盛，脾陽不振之證"
    },
    "淡紅色舌且舌苔淺白": {
        "體質": "健康舌相",
        "症狀": ["無明顯不適"],
        "建議": "保持現狀，預防為主",
        "食療": ["均衡飲食", "適量運動"],
        "禁忌": ["過度進補"],
        "中醫理論": "舌淡紅苔薄白為正常舌象，氣血調和之徵"
    }
}

class UserHealthManager:
    def __init__(self, data_file="users_health_data.json"):
        self.data_file = data_file
        self.users = {}
        self.load_data()

    def load_data(self):
        """載入用戶數據"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    self.users = json.load(f)
        except Exception as e:
            logging.error(f"載入用戶數據失敗: {str(e)}")
            self.users = {}

    def save_data(self):
        """保存用戶數據"""
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存用戶數據失敗: {str(e)}")

    def create_user(self, user_id: str, name: str = "", age: int = 0, gender: str = ""):
        """創建新用戶"""
        if user_id not in self.users:
            self.users[user_id] = {
                "name": name,
                "age": age,
                "gender": gender,
                "created_at": datetime.datetime.now().isoformat(),
                "tongue_records": [],
                "constitution_trends": {},
                "preferences": {}
            }
            self.save_data()
            logging.info(f"創建新用戶: {user_id}")
            return True
        return False

    def add_tongue_record(self, user_id: str, tongue_type: str, constitution: str, symptoms: List[str] = None):
        """添加舌象記錄"""
        if user_id not in self.users:
            self.create_user(user_id)

        record = {
            "record_id": str(uuid.uuid4()),
            "date": datetime.datetime.now().isoformat(),
            "tongue_type": tongue_type,
            "constitution": constitution,
            "symptoms": symptoms or [],
            "weather_info": f"記錄時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }

        self.users[user_id]["tongue_records"].append(record)
        self.update_constitution_trends(user_id, constitution)
        self.save_data()
        logging.info(f"添加舌象記錄: {user_id}, {tongue_type}")
        return record["record_id"]

    def update_constitution_trends(self, user_id: str, constitution: str):
        """更新體質趨勢統計"""
        trends = self.users[user_id].setdefault("constitution_trends", {})
        trends[constitution] = trends.get(constitution, 0) + 1

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """獲取用戶歷史記錄"""
        if user_id in self.users:
            records = self.users[user_id].get("tongue_records", [])
            return sorted(records, key=lambda x: x["date"], reverse=True)[:limit]
        return []

    def get_constitution_analysis(self, user_id: str) -> Dict:
        """分析用戶體質變化趨勢"""
        if user_id not in self.users:
            return {"error": "用戶不存在"}

        trends = self.users[user_id].get("constitution_trends", {})
        records = self.users[user_id].get("tongue_records", [])

        if not records:
            return {"message": "暫無記錄可供分析"}

        recent_constitution = records[-1]["constitution"]
        most_common = max(trends.items(), key=lambda item: item[1]) if trends else ("未知", 0)
        recent_records = records[-5:]
        constitutions = [r["constitution"] for r in recent_records]

        return {
            "recent_constitution": recent_constitution,
            "most_common_constitution": most_common[0],
            "constitution_frequency": trends,
            "recent_trend": constitutions,
            "total_records": len(records)
        }

# 初始化用戶管理器
user_manager = UserHealthManager()
current_user_id = "default_user"  # 將在主程序中動態設置

# === 工具函數 (Tool Functions) ===
@tool
def enhanced_tongue_analysis(input_str: str) -> str:
    """
    分析用戶提供的舌象描述和可選的伴隨症狀。
    輸入格式: '舌象描述' 或 '舌象描述|症狀1,症狀2'
    例如: '舌頭有紅點|口乾,失眠'
    """
    global current_user_id
    logging.info(f"執行舌象分析: {input_str} for user: {current_user_id}")
    if not input_str or ("|" not in input_str and input_str not in TONGUE_DATABASE):
        available_types = list(TONGUE_DATABASE.keys())
        return f"❌ 輸入格式錯誤或舌象類型無法識別。\n請輸入有效的舌象描述，例如：'舌頭有紅點' 或 '舌頭有紅點|口乾,失眠'\n\n📋 目前可分析的舌象類型：\n" + "\n".join([f"• {t}" for t in available_types])

    parts = input_str.split("|")
    tongue_type = parts[0].strip()
    symptoms = [s.strip() for s in parts[1].split(",")] if len(parts) > 1 and parts[1] else []

    tongue_info = TONGUE_DATABASE.get(tongue_type)
    if not tongue_info:
        available_types = list(TONGUE_DATABASE.keys())
        return f"❌ 未收錄此舌象類型：{tongue_type}\n\n📋 目前可分析的舌象類型：\n" + "\n".join([f"• {t}" for t in available_types])

    record_id = user_manager.add_tongue_record(
        current_user_id,
        tongue_type,
        tongue_info["體質"],
        symptoms
    )

    analysis = f"""
親愛的用戶，感謝您的信任。根據您提供的舌象描述，以下是專業分析與建議：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 分析編號：{record_id[:8]}...
👅 舌象描述：{tongue_type}
🏥 對應體質：{tongue_info["體質"]}

📚 【中醫理論基礎】
{tongue_info["中醫理論"]}

💊 【調理原則】
{tongue_info["建議"]}

🍵 【推薦食療方】
• {' • '.join(tongue_info["食療"])}

⚠️ 【飲食宜忌】
應避免：{' • '.join(tongue_info["禁忌"])}

🩺 【常見伴隨症狀】
• {' • '.join(tongue_info["症狀"])}

📝 【專業提醒】
此分析僅供參考，如症狀持續或加重，建議諮詢專業中醫師進行全面診斷。
"""

    if symptoms:
        matched_symptoms = [s for s in symptoms if any(s in expected for expected in tongue_info["症狀"])]
        unmatched_symptoms = [s for s in symptoms if not any(s in expected for expected in tongue_info["症狀"])]

        analysis += f"\n\n🎯 【症狀對照分析】"
        if matched_symptoms:
            analysis += f"\n✅ 符合體質特徵：{', '.join(matched_symptoms)}"
        if unmatched_symptoms:
            analysis += f"\n🔍 需進一步觀察：{', '.join(unmatched_symptoms)}"
            analysis += f"\n💡 這些症狀可能提示其他體質傾向，建議持續觀察記錄。請問您最近有無其他不適或想補充的症狀？"

    # 溫暖結尾與主動提示
    analysis += "\n\n📈 想了解您的健康趨勢，可輸入「健康趨勢」；查詢過往紀錄，請輸入「歷史記錄」。如需個性化建議，請輸入「個性化建議」。"
    return analysis

@tool
def get_user_health_trends(input_str: str) -> str:
    """獲取當前用戶的健康趨勢分析，無需輸入參數。"""
    global current_user_id
    logging.info(f"獲取健康趨勢: {current_user_id}")
    analysis = user_manager.get_constitution_analysis(current_user_id)

    if "error" in analysis:
        return analysis["error"]
    if "message" in analysis:
        return analysis["message"]

    trend_report = f"📈 【個人健康趨勢分析】\n━━━━━━━━━━━━━━━━━━━━\n"
    trend_report += f"🎯 當前體質: {analysis['recent_constitution']}\n"
    trend_report += f"🏆 主要體質: {analysis['most_common_constitution']}\n"
    trend_report += f"📊 總記錄數: {analysis['total_records']} 次\n\n"
    trend_report += "📋 【體質分布統計】\n"

    for constitution, count in analysis["constitution_frequency"].items():
        percentage = (count / analysis["total_records"]) * 100
        trend_report += f"• {constitution}: {count}次 ({percentage:.1f}%)\n"

    trend_report += f"\n🔄 【近期變化】\n最近{len(analysis['recent_trend'])}次記錄: {' → '.join(analysis['recent_trend'])}"
    return trend_report

@tool
def get_user_history_formatted(input_str: str) -> str:
    """獲取當前用戶最近10條的格式化歷史記錄，無需輸入參數。"""
    global current_user_id
    logging.info(f"獲取歷史記錄: {current_user_id}")
    history = user_manager.get_user_history(current_user_id, limit=10)

    if not history:
        return "📭 暫無歷史記錄"

    history_text = "📚 【歷史記錄】\n━━━━━━━━━━━━━━━━━━━━\n"
    for i, record in enumerate(history, 1):
        try:
            date = datetime.datetime.fromisoformat(record["date"]).strftime("%m/%d %H:%M")
        except (ValueError, TypeError):
            date = record["date"] # Fallback for old format
        history_text += f"{i:2d}. {date} | {record['tongue_type']} → {record['constitution']}\n"
    return history_text

@tool
def get_personalized_advice(input_str: str) -> str:
    """根據用戶的歷史記錄，生成個性化的中醫養生建議，無需輸入參數。"""
    global current_user_id
    logging.info(f"獲取個性化建議: {current_user_id}")
    analysis = user_manager.get_constitution_analysis(current_user_id)

    if "error" in analysis or "message" in analysis:
        return "請先進行至少一次舌象分析，以便為您提供個性化的中醫養生建議。"

    recent_constitution_key = next((k for k, v in TONGUE_DATABASE.items() if v['體質'] == analysis['recent_constitution']), None)
    most_common_key = next((k for k, v in TONGUE_DATABASE.items() if v['體質'] == analysis['most_common_constitution']), None)
    
    advice = f"🎯 【個人化中醫養生方案】\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    advice += f"基於您 {analysis['total_records']} 次舌診記錄的分析結果：\n\n"
    advice += f"🔸 目前主要體質：{analysis['recent_constitution']}\n"
    advice += f"🔸 長期體質傾向：{analysis['most_common_constitution']}\n\n"
    advice += "💫 【體質調理重點】\n"

    if most_common_key and most_common_key in TONGUE_DATABASE:
        constitution_info = TONGUE_DATABASE[most_common_key]
        advice += f"🌿 主要調理方向：{constitution_info['建議']}\n"
        advice += f"🍵 日常食療推薦：\n• {constitution_info['食療'][0]}：建議每日飲用\n• {constitution_info['食療'][1]}：每週2-3次\n"
        advice += f"🚫 飲食禁忌提醒：\n• {' • '.join(constitution_info['禁忌'])}\n"

    if analysis['recent_constitution'] != analysis['most_common_constitution'] and recent_constitution_key in TONGUE_DATABASE:
        recent_info = TONGUE_DATABASE[recent_constitution_key]
        advice += f"\n⚠️ 【近期體質變化提醒】\n您的體質最近偏向「{analysis['recent_constitution']}」，建議：\n"
        advice += f"• 臨時調整：{recent_info['建議']}\n"
        advice += f"• 近期適合：{', '.join(recent_info['食療'][:2])}\n"
        advice += f"• 暫時避免：{', '.join(recent_info['禁忌'][:2])}\n"
    
    advice += f"""
🕒 【養生時間建議】
• 最佳調理時間：每日清晨空腹時觀察舌象
• 食療頻率：建議持續3-4週觀察效果
• 複診提醒：建議每週記錄1-2次舌象變化

🌟 【中醫養生要點】
「藥補不如食補，食補不如睡補」
體質調理需要耐心，建議配合規律作息和適量運動效果更佳。
"""
    return advice


@tool
def weather_constitution_advice(city: str) -> str:
    """
    結合用戶體質和指定城市當前天氣，提供中醫調理建議。
    輸入格式: '城市名稱'
    例如: '台北'
    """
    global current_user_id
    logging.info(f"獲取天氣調理建議: {city} for user: {current_user_id}")
    if not OPENWEATHER_API_KEY:
        return "❌ 缺少 OpenWeather API 金鑰，無法查詢天氣。"

    # 新增：常見台灣城市中文到英文自動轉換
    CITY_NAME_MAP = {
        "台北": "Taipei",
        "臺北": "Taipei",
        "新北": "New Taipei",
        "台中": "Taichung",
        "臺中": "Taichung",
        "台南": "Tainan",
        "臺南": "Tainan",
        "高雄": "Kaohsiung",
        "基隆": "Keelung",
        "桃園": "Taoyuan",
        "新竹": "Hsinchu",
        "嘉義": "Chiayi",
        "屏東": "Pingtung",
        "宜蘭": "Yilan",
        "花蓮": "Hualien",
        "台東": "Taitung",
        "臺東": "Taitung",
        "南投": "Nantou",
        "彰化": "Changhua",
        "雲林": "Yunlin",
        "苗栗": "Miaoli",
        "澎湖": "Penghu",
        "金門": "Kinmen",
        "連江": "Lienchiang",
    }
    city_query = CITY_NAME_MAP.get(city.strip(), city.strip())

    @lru_cache(maxsize=32)
    def get_weather_cached(city_name: str) -> Dict:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}&units=metric&lang=zh_tw"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"天氣查詢失敗: {str(e)}")
            return {"error": str(e)}

    res = get_weather_cached(city_query)
    if res.get("cod") != 200:
        return f"⚠️ 天氣資訊查詢失敗：{res.get('message', '未知錯誤')}\n請檢查城市名稱或網路連線狀態"

    humidity = res['main']['humidity']
    temperature = res['main']['temp']
    desc = res['weather'][0]['description']

    analysis = user_manager.get_constitution_analysis(current_user_id)
    constitution = analysis.get("recent_constitution", "未知")

    weather_advice = f"🌤️ 【天人合一調理方案】\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    weather_advice += f"📍 地點：{city}\n🌡️ 溫度：{temperature}°C\n💧 濕度：{humidity}%\n☁️ 天候：{desc}\n👤 您的體質：{constitution}\n\n"
    weather_advice += "🍃 【中醫天氣養生理論】\n中醫認為人體應順應天時變化，「天人合一」方能保持健康。\n"

    if humidity > 75:
        weather_advice += f"\n💧 【濕氣調理】\n當前濕度偏高（{humidity}%），易助長體內濕邪\n• 通用建議：多食祛濕食物如薏仁、冬瓜、赤小豆\n• 避免：冰冷飲品、甜膩食物，以免助濕生痰\n"
        if constitution == "脾虛濕重":
            weather_advice += "• 您的體質特別容易受濕氣影響，建議加強健脾祛濕，推薦四神湯、茯苓茶。\n"
        elif constitution == "濕熱體質":
            weather_advice += "• 濕熱體質在高濕環境下需特別注意清熱利濕，推薦綠豆薏仁湯、荷葉茶。\n"
    
    if temperature > 30:
        weather_advice += f"\n☀️ 【暑熱調理】\n高溫天氣（{temperature}°C），需注意防暑清熱\n• 建議：綠豆湯、酸梅湯等清暑飲品\n• 避免：過度貪涼，以免寒熱夾雜\n"
    elif temperature < 15:
        weather_advice += f"\n❄️ 【寒冷調理】\n低溫環境（{temperature}°C），需注意溫陽暖身\n• 建議：生薑茶、桂圓茶等溫性飲品\n• 避免：過食生冷，注意保暖防寒\n"

    weather_advice += "\n💡 中醫智慧：「順天時而動，應地利而食，合人和而居」"
    return weather_advice

# === 工具與 Agent 設置 (Tools and Agent Setup) ===
tools = [
    enhanced_tongue_analysis,
    get_user_health_trends,
    get_user_history_formatted,
    get_personalized_advice,
    weather_constitution_advice
]

# **修正**: 移除 'messages_modifier' 參數
agent_executor = create_react_agent(llm, tools)

# === 主程序 (Main Program) ===
async def main():
    global current_user_id
    print("🚀 正在初始化中醫舌象智能分析系統...")
    user_id_input = input("請輸入您的用戶 ID（或按 Enter 使用 'default_user'）：").strip()
    current_user_id = user_id_input or "default_user"
    
    if user_manager.create_user(current_user_id):
        print(f"歡迎新用戶 {current_user_id}！")
    else:
        print(f"歡迎回來，{current_user_id}！")

    print("\n🏥 中醫舌象智能分析系統 v2.2 啟動！")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("💡 使用說明:")
    print("1. 舌象分析: 直接描述您的舌象，如 '舌頭有紅點'")
    print("   或附加症狀: '舌頭有紅點|口乾,失眠'")
    print("2. 查看趨勢: 輸入 '健康趨勢'")
    print("3. 查看歷史: 輸入 '歷史記錄'")
    print("4. 個性建議: 輸入 '個性化建議'")
    print("5. 天氣調理: 輸入 '天氣 台北'")
    print("6. 輸入 'q' 或 'quit' 退出系統")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    chat_history = []

    while True:
        user_input = input("🧑 您: ").strip()

        if user_input.lower() in ['q', 'quit']:
            print("👋 感謝使用中醫舌象分析系統，祝您身體健康！")
            break

        if not user_input:
            continue

        # === 新增：明確指令直接呼叫工具，提升穩定性 ===
        if user_input in ["健康趨勢"]:
            final_answer = get_user_health_trends("")
            print(f"🤖 AI中醫師: {final_answer}\n")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            continue
        if user_input in ["歷史記錄"]:
            final_answer = get_user_history_formatted("")
            print(f"🤖 AI中醫師: {final_answer}\n")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            continue
        if user_input in ["個性化建議"]:
            final_answer = get_personalized_advice("")
            print(f"🤖 AI中醫師: {final_answer}\n")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            continue
        if user_input.startswith("天氣"):
            city = user_input.replace("天氣", "").strip()
            if not city:
                print("請輸入城市名稱，例如：天氣 台北\n")
                continue
            final_answer = weather_constitution_advice(city)
            print(f"🤖 AI中醫師: {final_answer}\n")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            continue

        # === 新增：舌象描述直接呼叫分析工具 ===
        tongue_types = list(TONGUE_DATABASE.keys())
        if any(user_input.startswith(t) for t in tongue_types):
            final_answer = enhanced_tongue_analysis(user_input)
            print(f"🤖 AI中醫師: {final_answer}\n")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            continue

        try:
            # **修正**: 手動將 System Prompt 添加到 message 列表中
            messages_with_system_prompt = [
                SystemMessage(content=TCM_SYSTEM_PROMPT)
            ] + chat_history + [HumanMessage(content=user_input)]
            
            response = agent_executor.invoke({
                "messages": messages_with_system_prompt
            })
            
            final_answer = response['messages'][-1].content
            print(f"🤖 AI中醫師: {final_answer}\n")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except Exception as e:
            error_message = f"❌ 系統發生錯誤: {str(e)}"
            print(error_message)
            logging.error(f"Agent 執行錯誤: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
