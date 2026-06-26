# src/graph/state.py
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# 从原代码中提取的常量和枚举
class ChatAction(Enum):
    HORNY_DETECT = "horny_detect"
    CREATOR_PROFILE = "creator_profile"
    FAN_PROFILE = "fan_profile"
    ACTIVATE = "activate"
    PPV_ACTIVATE = "ppv_activate"
    CASUAL_TEASING = "casual_teasing"
    SEXTING = "sexting"
    BURNING = "burning"
    TASK = "task"
    PPV_FOLLOWUP = "ppv_follow_up"
    CHAT_FOLLOWUP = "chat_follow_up"
    PPV_CHAT = "ppv_chat"
    PPV_TEASING_SEXTING = "ppv_teasing_sexting"
    PPV_FAQ_CR = "ppv_faq_cr"
    PROFILE_SUMMARY = "profile_summary"
    VIOLATION_DETECT = "violation_detect"
    HALLUCINATION_DETECT = "hallucination_detect"
    MEDIA_TAG = "media_tag"
    FAN_INTENTION = "fan_intention"
    FAN_INTENTION_CONTENT_REQUEST = "fan_intention_content_request"
    FAN_INTENTION_READY_TO_PURCHASE = "fan_intention_ready_to_purchase"
    FAN_INTENTION_BOND = "fan_intention_bond"
    FAN_INTENTION_TEASE = "fan_intention_tease"
    FAN_INTENTION_SEXTING = "fan_intention_sexting"
    FAN_INTENTION_OTHER = "fan_intention_other"
    FAN_INTENTION_PREVIEW_REQUEST = "fan_intention_preview_request"
    FAN_INTENTION_NEGOTIATION = "fan_intention_negotiation"
    FAN_INTENTION_OBJECTION = "fan_intention_objection"
    FAN_INTENTION_PPV_INQUIRY = "fan_intention_ppv_inquiry"
    PPV_REQUEST_REFUSE = "ppv_request_refuse"

    # ========== 新增A/B测试相关字段 ==========
    ab_test_task_id: str = "model_quality_test_001"  # 默认A/B测试任务ID
    ab_test_group: str = "control_group"  # A/B分组
    model_config: Dict[str, Any] = Field(default_factory=dict)  # 分组对应的模型配置

# 幻觉检测步骤日志（用于可观测性）
class HallucinationStepLog(BaseModel):
    step_name: str
    is_hit: bool  # 是否命中该步骤（触发幻觉判定/跳过逻辑）
    cost_time: float  # 步骤耗时（秒）
    message: str  # 步骤说明
    error: Optional[str] = None  # 步骤异常信息

# 幻觉检测专用状态（可嵌入到你的核心 AgentState 中）
class HallucinationDetectionState(BaseModel):
    # 输入：必须由上游节点（如 ChatGenerator）传入
    context: Optional[Dict[str, Any]] = None  # 对话上下文
    generated_response: Optional[str] = None  # 生成的自动回复

    # 输出：幻觉检测结果
    has_hallucination: Optional[bool] = None  # 是否存在幻觉
    hallucination_step_logs: List[HallucinationStepLog] = []  # 步骤日志

    # 中间状态：避免重复计算
    ppv_sent_status: Optional[bool] = None  # PPV 是否已发送
    pure_emoji_check: Optional[bool] = None  # 是否为纯表情
    regex_hit_check: Optional[bool] = None  # 是否匹配正则规则

class MatchDegree(Enum):
    STRONG = "strong"
    WEAK = "weak"
    AMBIGUOUS = "ambiguous"


class RecommendFailReason(Enum):
    UNSUPPORTED_TYPE = "The creator doesn't support the type of ppv which fan asked"
    NO_CANDIDATE = "No available candidate PPV for the query"
    WEAK_MATCH = "The ppv weakly match the query of fan"


class SubscribeVIPTask:
    NOT_COMPLETE = 0
    HIGH_HORNY = 1
    TIP = 2
    CONTENT_REQUEST = 3
    MEDIA = 4


class PpvMaterial(BaseModel):
    id: str
    type: int


class BundleItem(BaseModel):
    material_id: str | None = None
    material_type: int | None = None


class Bundle(BaseModel):
    bundle_id: int | None = None
    reference_price: int | None = None
    materials: List[BundleItem] | None = None

    @property
    def exists(self):
        return self.materials and len(self.materials) > 1


class RecommendMaterial(BaseModel):
    material_id: str | None = None
    tags: List[str] | None = None
    score: float | None = None
    reference_price: int | None = None
    material_type: int | None = None
    query_match_degree: MatchDegree = MatchDegree.AMBIGUOUS
    recall_method: str | None = None
    bundle: Bundle | None = None


class FanQueryTag(BaseModel):
    category: Dict[str, float] = Field(default_factory=dict)
    scene: Dict[str, float] = Field(default_factory=dict)
    participants: Dict[str, float] = Field(default_factory=dict)
    body_parts: Dict[str, float] = Field(default_factory=dict)
    clothing: Dict[str, float] = Field(default_factory=dict)
    action: Dict[str, float] = Field(default_factory=dict)
    props: Dict[str, float] = Field(default_factory=dict)


class DialogueState(BaseModel):
    intention: str | None = None
    action: str | None = None
    task_completed_reason: int | None = SubscribeVIPTask.NOT_COMPLETE
    horny_value: float = -1.0  # DEFAULT_HORNY_VALUE
    response: str | None = None
    fan_query_tags: FanQueryTag = Field(default_factory=FanQueryTag)
    recommend_material: List[RecommendMaterial] = Field(default_factory=list)
    recommend_fail_reason: RecommendFailReason | None = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of DialogueState")


class ChatMessage(BaseModel):
    role: Literal["system", "fan", "creator"]
    content: str | None = None
    type: str = "text"
    language: str = "en"
    message_id: str | None = None


class MessageItem(BaseModel):
    role: Literal["system", "fan", "creator"]
    message_id: str | None = None
    content: List[Dict[str, Any]] = Field(default_factory=list)


class HistoryContentItem(BaseModel):
    type: str
    content: Optional[str] = None
    price: Optional[int] = None
    bundle_id: Optional[int] = None
    materials: Optional[List[PpvMaterial]] = None


class ChatHistory(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

    def get_flatten_history(self) -> List[Dict[str, str]]:
        return self.messages

    def get_last_fan_question(self) -> ChatMessage:
        for msg in reversed(self.messages):
            if msg["role"] == "fan":
                return ChatMessage(role="fan", content=msg["content"])
        return ChatMessage(role="fan", content="")

    def get_all_history(self) -> List[Dict[str, str]]:
        return self.messages


class BusinessInfo(BaseModel):
    creator_profile: Dict[str, Any] = Field(default_factory=dict)
    fan_profile: Dict[str, Any] = Field(default_factory=dict)
    scene: str = "chat"
    summary: Dict[str, Any] = Field(default_factory=dict)
    input_interrupted: bool = False
    interrupted_reason: str = ""
    task_complete_reason: int = SubscribeVIPTask.NOT_COMPLETE


class AgentState(BaseModel):
    """LangGraph状态定义"""
    request_id: str
    message: ChatMessage
    message_item: Optional[MessageItem] = None
    chat_history: ChatHistory
    dialog_states: List[DialogueState] = Field(default_factory=list)
    dialog_state: DialogueState = Field(default_factory=DialogueState)
    creator_profile: Dict[str, Any] = Field(default_factory=dict)
    fan_profile: Dict[str, Any] = Field(default_factory=dict)
    creator_script: List[str] = Field(default_factory=list)
    record_data: Dict[str, Any] = Field(default_factory=dict)
    mode: str | None = None
    scene: str | None = None
    status: int = 0
    business_info: BusinessInfo = Field(default_factory=BusinessInfo)

    # LangGraph特定字段
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    should_regenerate: bool = False
    retry_count: int = 0
    max_retry: int = 2
    final_response: str = ""
    error_message: str = ""
    handoff_required: bool = False
    handoff_reason: str = ""
    risk_level: str = "low"
    trace_id: str | None = None
    task_completed: bool = False
    subscribe_vip_task: int = SubscribeVIPTask.NOT_COMPLETE
    ai_interrupted: bool = False
    interrupted_reason: str = ""
    fan_profile_summary: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.trace_id:
            self.trace_id = self.request_id

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Context")

    def add_record_data(self, key: str, value: Any):
        """添加记录数据"""
        self.record_data[key] = value

    def get_formatted_record_data(self) -> dict[str, Any]:
        """获取格式化后的记录数据"""
        record = {
            "time": datetime.now().isoformat(timespec='seconds'),
            "fan_profile_summary": self.record_data.get("fan_profile_summary", "{}"),
            "subscribe_vip_task": self.record_data.get("subscribe_vip_task"),
            "ai_interrupted": self.record_data.get("ai_interrupted"),
            "interrupted_reason": self.record_data.get("interrupted_reason", ""),
            "horny_value": self.dialog_state.horny_value,
        }

        # 收集指定字段到extensions中
        EXTENSION_FIELDS = ["ppv_rec_ids", "fan_query_tags", "ppv_rec_candidate_detail", "fan_intention", "action",
                            "has_paid_link", "fan_profile_memory", "ppv_rec_bundle_detail"]
        extensions = {}
        for key in EXTENSION_FIELDS:
            if key in self.record_data:
                extensions[key] = self.record_data[key]

        # Remove empty values
        extensions = {k: v for k, v in extensions.items() if v is not None and v != {} and v != [] and v != ""}
        if extensions:
            record["extensions"] = extensions

        return record


    def get_formatted_record_data(self) -> Dict:
        # 原有逻辑...
        record = {
            # 原有字段...
            "request_id": self.request_id,
            "scene": self.scene,
            "faq_matched": self.faq_matched,
            # ========== 新增A/B测试记录 ==========
            "ab_test_task_id": self.ab_test_task_id,
            "ab_test_group": self.ab_test_group,
            "model_name": self.model_config.get("model_name", "unknown")
        }
        return record
