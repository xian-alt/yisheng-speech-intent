"""Protocol shared by training, inference and API."""

POLICY = {
    "REQUEST_WATER": ("normal", "caregiver", "老人请求饮水"),
    "REQUEST_FOOD": ("normal", "caregiver", "老人请求用餐"),
    "REQUEST_TOILET": ("high", "caregiver", "老人需要如厕协助"),
    "CALL_CAREGIVER": ("high", "caregiver", "老人呼叫护工"),
    "CONTACT_FAMILY": ("normal", "family", "老人希望联系家属"),
    "MEDICINE_REMINDER": ("review", "caregiver", "用药需求待人工确认"),
    "SOS": ("urgent", "caregiver", "紧急求助"),
    "LIGHT_ON": ("normal", "caregiver", "老人请求开灯"),
    "LIGHT_OFF": ("normal", "caregiver", "老人请求关灯"),
    "TV_ON": ("normal", "caregiver", "老人请求打开电视"),
    "TV_OFF": ("normal", "caregiver", "老人请求关闭电视"),
    "ADJUST_TEMPERATURE": ("normal", "caregiver", "老人请求调节温度"),
    "UNKNOWN": ("review", "caregiver", "请人工核查原话"),
}
INTENTS = set(POLICY)
ALIASES = dict(
    zip(
        (
            "request_water",
            "request_food",
            "request_toilet",
            "request_help",
            "contact_family",
            "request_medicine",
            "emergency_call",
            "request_light_on",
            "request_light_off",
            "request_tv_on",
            "request_tv_off",
            "request_temperature_adjust",
            "unknown",
        ),
        POLICY,
    )
)
INSTRUCTION_V1 = (
    "判断老人原话的当前服务意图，只输出一个意图标签。标签："
    + ", ".join(POLICY)
    + "。否定需求、闲聊、历史事件或无法理解时输出 UNKNOWN。当前摔倒、紧急求救输出 SOS。"
)
INSTRUCTION = (
    '判断老人原话的当前有效服务需求，只输出一个意图标签。标签：'
    + ', '.join(POLICY)
    + '。具体目的优先于陪同方式，例如陪我去卫生间输出REQUEST_TOILET；'
      '只要求有人来陪伴输出CALL_CAREGIVER。否定的需求不算当前请求，'
      '改口后仍有效的需求应正确分类，例如不喝水了想吃饭输出REQUEST_FOOD。'
      '闲聊、转述、仅描述历史事件或无法理解输出UNKNOWN；当前摔倒、紧急求救输出SOS。'
      '多个独立同时有效的需求不能由一个标签完整表示时输出UNKNOWN，留待人工确认。'
)
INSTRUCTION_VERSION = 'current-purpose-v2'


def validated_intent(value):
    if isinstance(value, str):
        value = {"intent": value}
    if not isinstance(value, dict):
        value = {}
    label = value.get("intent")
    label = ALIASES.get(label, label) if isinstance(label, str) else "UNKNOWN"
    if label not in POLICY:
        label = "UNKNOWN"
    priority, target, message = POLICY[label]
    params = value.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    clean = {}
    if label == "ADJUST_TEMPERATURE":
        temp = params.get("temperature")
        if (
            isinstance(temp, (int, float))
            and not isinstance(temp, bool)
            and 16 <= temp <= 30
        ):
            clean["temperature"] = temp
        if params.get("direction") in ("warmer", "cooler"):
            clean["direction"] = params["direction"]
    return dict(
        intent=label,
        priority=priority,
        target=target,
        message=message,
        parameters=clean,
    )
