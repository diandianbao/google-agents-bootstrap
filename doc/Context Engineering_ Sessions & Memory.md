# 记忆检索（接上文）
要搞定“哪些记忆该用、啥时候用”，得先看记忆是咋整理的：要是像“用户档案”那样整得规规矩矩，直接查全量档案或者某一项信息就行；可要是一堆零散的记忆，检索就复杂多了——得从一大堆没咋规整的信息里，找出跟当前对话最相关、最有用的内容，这才是真正的难点。而且还得在保证速度的前提下，挑出“真有用”的记忆，要是给模型塞了没用的记忆，反而会让回复变糟；但要是刚好找到关键信息，就能让交互一下子变智能。

高级点的记忆系统，不只会简单搜一搜，还会从好几个维度给记忆打分，挑出最合适的：
- **相关性（语义相似度）**：这个记忆跟当前对话在意思上贴得多近？
- **新鲜度（时间维度）**：这个记忆是啥时候创建的？越新的往往越有用。
- **重要性（关键程度）**：这个记忆本身有多重要？跟相关性不一样，“重要性”一般在生成记忆的时候就定下来了。

很多人容易犯的错就是“只看语义相似度”。有时候看着意思像，但其实是老早以前的、无关紧要的记忆，根本帮不上忙。所以最好的办法是“多维度结合”，把这三个维度的分数揉在一起算。

要是对准确性要求特别高，还能搞点进阶操作，比如改写查询词、重新排序，或者用专门的检索工具。但这些操作都特费劲儿，还会拖慢速度，不适合大部分实时场景。要是实在需要这些复杂操作，而且记忆不会很快过期，那可以整个“缓存”——把检索结果暂时存起来，下次再查一样的内容，就不用再费时间算了。

比如“改写查询词”，可以让大模型把用户模糊的问题改得更精准，或者把一个问题拆成好几个相关的小问题，覆盖更多角度。这样能让搜索结果更准，但开头得多调用一次大模型，会慢一点。

“重新排序”就是先通过相似度搜索，找出一大波候选记忆（比如前50个），再让大模型重新评估、排序这一小批，最后挑出更准的。

还有种方法是“微调专门的检索工具”，但这得有标注好的数据，成本也高，一般用不上。

说到底，最好的检索方法，其实是从“生成记忆”就开始下功夫。只要记忆库里存的都是高质量、有用的信息，不管咋检索，拿到的记忆都差不了。

## 检索时机
最后一个要定的架构问题是“啥时候检索记忆”，主要有两种方式：

### 1. 主动检索（每轮对话都加载）
每次对话一开始，就自动把记忆加载进来。这样能保证随时有上下文可用，但要是某轮对话用不上记忆，就白浪费时间了。不过好在单轮对话里，记忆是不变的，所以可以用缓存来提速，减少这种浪费。

比如用ADK框架的话，能直接用内置的“预加载记忆工具”，或者自己写个回调函数来实现：
```python
# 方法1：用内置的PreloadMemoryTool，每轮都通过相似度搜索拉取记忆
agent = LlmAgent(
    tools=[adk.tools.preload_memory_tool.PreloadMemoryTool()]
) 

# 方法2：自己写回调函数，更灵活地控制怎么拉取记忆
def retrieve_memories_callback(callback_context, llm_request):
    # 拿到当前用户ID和应用名
    user_id = callback_context._invocation_context.user_id
    app_name = callback_context._invocation_context.app_name
    
    # 调用API拉取记忆
    response = client.agent_engines.memories.retrieve(
        name="projects/你的项目ID/locations/你的区域/reasoningEngines/你的引擎名", 
        scope={
            "user_id": user_id, 
            "app_name": app_name
        }
    )
    
    # 把记忆整理成列表
    memories = [f"* {memory.memory.fact}" for memory in list(response)]
    # 要是没记忆，就不往系统指令里加了
    if not memories:
        return
    
    # 把记忆拼到系统指令里
    llm_request.config.system_instruction += "\n以下是你掌握的用户信息：\n"
    llm_request.config.system_instruction += "\n".join(memories) 

# 创建智能体，把回调函数加上
agent = LlmAgent(
    before_model_callback=retrieve_memories_callback,
)
```

### 2. 被动检索（“记忆即工具”）
给智能体整个“查记忆”的工具（比如叫`load_memory`），让它自己判断啥时候需要查。这种方式更高效、更靠谱，但得多调用一次大模型，会慢一点、花点钱；不过好处是“只在需要的时候查”，不会平白浪费时间。另外，智能体可能不知道有没有相关记忆，这时候可以在工具说明里告诉它“有哪些类型的记忆”，比如“存了用户喜欢的食物这类信息”，帮它做判断。

```python
# 方法1：用内置的LoadMemory工具
agent = LlmAgent(
    tools=[adk.tools.load_memory_tool.LoadMemoryTool()],
)

# 方法2：自己写工具，说明清楚有哪些记忆可以查
def load_memory(query: str, tool_context: ToolContext):
    """帮用户拉取记忆。
    目前存的用户信息包括：
    * 用户偏好，比如喜欢的食物
    """
    # 通过相似度搜索拉取记忆
    response = tool_context.search_memory(query)
    return response.memories

# 创建智能体，把自定义工具加上
agent = LlmAgent(
    tools=[load_memory],
)
```

# 基于记忆的推理
找到相关记忆后，最后一步就是“怎么把记忆塞到大模型的上下文窗口里”——这步特别关键，记忆放哪儿、怎么放，会严重影响大模型的推理，还会关系到成本和回复质量。

一般有两种主要方式：把记忆贴在系统指令后面，或者塞到对话历史里。实际用的时候，往往是“两种结合”：系统指令里放那些稳定的、全局通用的记忆（比如用户档案），保证每次都有；对话历史里放那些临时的、只跟当前对话相关的记忆（比如之前聊过的某个话题细节），灵活应对当下需求。

## 系统指令中的记忆
最简单的办法就是把记忆追加到系统指令里。这样能保持对话历史干净，直接把记忆当成“基础上下文”，跟系统指令拼在一起。比如用Jinja模板动态加记忆：

```python
from jinja2 import Template

# 写个模板，把系统指令和记忆拼起来
template = Template("""
{{ system_instructions }}
<记忆内容>
以下是关于当前用户的信息：
{% for retrieved_memory in data %}* {{ retrieved_memory.memory.fact }}
{% endfor %}
</记忆内容>
""")

# 把实际的系统指令和记忆填进模板
prompt = template.render(
    system_instructions=system_instructions,
    data=retrieved_memories
)
```

这种方式有三个好处：记忆的“权重”高，大模型会更重视；能把记忆和对话内容清楚分开；特别适合用户档案这种稳定的信息。但也有缺点——可能会“过度影响”大模型，比如不管聊啥，它都硬要往记忆上靠，哪怕不相关。

另外还有几个限制：首先，得要智能体框架支持“每调用一次大模型就动态改系统指令”，不是所有框架都有这功能；其次，跟“记忆即工具”不兼容——因为得先确定系统指令，大模型才能判断要不要调用查记忆的工具；最后，没法处理非文本记忆，大部分大模型的系统指令只认文字，图片、音频这类内容塞不进去。

## 对话历史中的记忆
这种方式是把记忆直接插进逐轮对话里，可以放在整个对话历史前面，也可以放在最新的用户提问前面。

但问题也很明显：会让对话变乱，增加token消耗；要是记忆不相关，还会让大模型迷糊。最大的风险是“对话注入”——大模型可能会误以为记忆是“之前实际说过的话”。而且插记忆的时候，得注意“视角”，比如要是用“用户”的角色，那记忆就得用第一人称写，比如“我喜欢靠窗的座位”，而不是“用户喜欢靠窗的座位”。

还有种特殊情况：通过工具调用拉取记忆。这时候记忆会作为“工具输出”，直接出现在对话里。

```python
def load_memory(query: str, tool_context: ToolContext):
    """把记忆加载到对话历史里"""
    # 调用工具拉取记忆
    response = tool_context.search_memory(query)
    return response.memories

# 创建智能体，添加这个工具
agent = LlmAgent(
    tools=[load_memory],
)
```

## 过程性记忆
前面咱们聊的基本都是“陈述性记忆”（也就是“知道是什么”），这也跟现在市面上大部分记忆工具的定位一致——它们擅长提取、存储“事实、历史、用户数据”这类信息。

但这些工具没法处理“过程性记忆”（也就是“知道怎么做”）——这种记忆是用来优化智能体的工作流程和推理方式的。存储“怎么做”不是“找信息”的问题，而是“怎么帮智能体更好地思考”的问题。要管好这种记忆，得有一套完全独立的、专门的流程，不过整体框架跟陈述性记忆差不多：

1. **提取**：得用专门的提示词，从成功的交互里提炼出“可复用的步骤”（比如“订机票的流程”），而不只是抓个事实。
2. **整合**：陈述性记忆的整合是“合并相关事实”，但过程性记忆是“优化流程本身”——比如把新的好方法跟现有的“最佳步骤”结合，修补老流程里的漏洞，删掉没用的步骤。
3. **检索**：目的不是“找数据回答问题”，而是“找一套步骤指导智能体做复杂任务”，所以过程性记忆的格式可能跟陈述性记忆不一样。

这种“智能体自己优化流程”的能力，很容易让人想到“微调”（比如基于人类反馈的强化学习RLHF）。但两者差别很大：微调是慢节奏的离线训练，会改模型的权重；而过程性记忆是“实时优化”——直接把正确的“步骤指南”塞到提示词里，通过“上下文学习”帮智能体干活，不用微调。

# 测试与评估
搞出带记忆功能的智能体后，得通过全面的测试来验证它的表现。评估得从好几个层面来：智能体是不是记了该记的（记忆质量）、要用的时候能不能找到（检索效果）、用了记忆之后能不能更好地完成任务（任务成功率）。学术界喜欢搞可重复的基准测试，但工业界更关注“记忆对实际生产环境中智能体的性能和可用性有啥影响”。

## 1. 记忆生成质量指标
看记忆本身好不好，核心是“智能体有没有记对东西”。一般会把智能体生成的记忆，跟人工整理的“标准答案记忆”对比：
- **精确率**：智能体生成的所有记忆里，有多少是准确、相关的？精确率高，能避免记忆库被没用的信息塞满。
- **召回率**：从对话里该记的关键信息中，智能体抓住了多少？召回率高，能保证不遗漏重要内容。
- **F1分数**：精确率和召回率的调和平均数，用来综合衡量记忆质量。

## 2. 记忆检索性能指标
看智能体能不能在需要的时候找到正确的记忆：
- **Recall@K（前K召回率）**：需要某段记忆时，它能不能出现在前K个检索结果里？这是判断检索系统准不准的核心指标。
- **延迟**：检索是用户交互的“关键路径”，整个过程必须快（比如200毫秒以内），不然会影响用户体验。

## 3. 端到端任务成功率指标
这是最终的测试——“记忆到底能不能帮智能体把活干得更好？”。一般会让智能体用记忆完成任务，再让另一个大模型当“裁判”，把智能体的输出和“标准答案”对比，判断准不准，以此衡量记忆系统的贡献。

评估不是搞一次就完了，而是“持续优化的循环”：先定个基准，分析哪里不行，调整系统（比如改提示词、优化检索算法），再重新评估看效果。

除了质量，生产环境还得关注性能：每个评估环节都要测算法的延迟，以及能不能扛住高并发。检索是“关键路径”，得保证秒级以内的速度；生成和整合虽然一般是异步处理，但也得有足够的吞吐量，能跟上用户需求。说到底，好的记忆系统既要“聪明”，又要“高效、靠谱”，能应对真实场景。

# 记忆的生产环境考量
把带记忆的智能体从原型搞到生产环境，除了性能，还得关注企业级的架构问题——可扩展性、容错能力、安全性都得达标。生产级的系统不光要“智能”，还得“稳”。

要保证用户体验不被“生成记忆”这种费劲儿的操作拖慢，关键是把“记忆处理”和“主应用逻辑”拆开。虽然这是个事件驱动的模式，但一般不用自己搭消息队列，直接调用专门的记忆服务API就行，而且要“非阻塞”——具体流程是这样的：

1. **智能体推数据**：比如对话结束后，智能体调用记忆服务的API，把原始数据（比如对话记录）“推”过去，不用等结果。
2. **记忆服务后台处理**：记忆服务马上确认“收到了”，然后把生成任务放进自己的队列里，异步处理——比如调用大模型提取、整合记忆。有时候还会等用户隔一会儿不操作了再处理，避免频繁触发。
3. **记忆持久化存储**：服务把最终的记忆（可能是新的，也可能是更新旧的）存到专门的数据库里。要是用托管的记忆服务，存储都是现成的。
4. **智能体拉取记忆**：下次用户交互需要上下文时，智能体直接查这个记忆库就行。

这种“基于服务的非阻塞方式”有个大好处：就算记忆处理出问题或者变慢，也不会影响用户用应用，系统更抗造。另外，还能选“实时生成”（适合保证对话新鲜度）或者“离线批量处理”（适合用历史数据初始化系统）。

随着应用用户变多，记忆系统得能扛住高频请求，还不能出错。比如多个请求同时要改同一个记忆，得防止“死锁”或者“竞争冲突”——可以用数据库的事务操作，或者“乐观锁”来解决，但这可能会导致请求排队。所以还得整个靠谱的消息队列，缓冲大量请求，别让记忆服务被压垮。

记忆服务还得能应对“临时错误”：比如调用大模型失败了，得有“重试机制”（比如失败后等一会儿再试，间隔越变越长），要是一直失败，就放到“死信队列”里，之后分析原因。

要是应用是全球用的，记忆服务的数据库得支持“多区域同步”，保证速度快、不宕机。不能在客户端同步数据，因为整合记忆需要“数据一致”，不然会冲突。所以记忆系统得自己处理同步，对外只给开发者一个“统一的数据库接口”，但背后要保证全球数据一致。

像“Agent Engine Memory Bank”这种托管记忆服务，会帮你搞定这些生产级的问题，你只用专注于智能体的核心逻辑就行。

# 隐私与安全风险
记忆里都是用户数据，所以隐私和安全控制必须严。可以这么理解：记忆系统就像公司里的“保密档案室”，管理员（对应系统）的活儿就是“存有用的信息，同时护好安全”。

这个档案室的“铁规矩”是“数据隔离”——就像管理员不会把不同部门的机密文件混在一起，记忆也得按用户或者租户严格分开。给A用户服务的智能体，绝对不能拿到B用户的记忆，这得靠严格的访问控制列表（ACL）来实现。另外，用户得能自己控制数据，比如可以关了记忆生成，或者要求删除所有自己的记忆。

管理员存文件前，还得做两件关键的安全操作：第一，把敏感的个人信息（PII）删掉，比如手机号、邮箱，这样既存了有用的信息，又不会有泄露风险；第二，得识别出假的、骗人的信息，防止有人故意塞坏数据（也就是“记忆投毒”）。同理，系统在把信息存成长期记忆前，也得验证、清理，比如用Model Armor这种工具，防止恶意用户通过提示词注入搞坏智能体的记忆。

还有个“信息泄露”的风险：要是多个用户共享一套记忆（比如过程性记忆，教智能体怎么做某事），比如把A用户的流程当例子给B用户看，得先把敏感信息完全去掉，不然会跨用户泄露数据。

# 结论
这份白皮书聊了“上下文工程”这个领域，重点讲了它的两个核心部分：对话会话（Sessions）和记忆（Memory）。从简单的一轮对话，到能长期用、能落地的智能信息，全靠上下文工程——它要把对话历史、记忆、外部知识这些所有需要的信息，动态整合成大模型能用上的上下文。而这一切，都依赖“会话”和“记忆”这两个既独立又关联的系统。

会话管的是“当下”——它是单次对话的“临时容器”，得快、得安全：访问速度要快，用户才觉得流畅；数据要严格隔离，不能串了。为了防止上下文窗口不够用、拖慢速度，还得用“按token截断”“递归总结”这些方法，压缩会话里的内容。另外，安全方面，会话数据存之前，必须把敏感个人信息（PII）删掉。

记忆则是“长期个性化”的核心，也是跨会话持久化信息的关键。它跟RAG不一样——RAG让智能体成了“事实专家”，而记忆让智能体成了“用户专家”。记忆是个“大模型驱动的主动ETL流程”，要做提取、整合、检索这三件事，从对话里提炼最重要的信息：提取阶段把关键信息抓出来，整合成记忆；整合阶段把新记忆跟旧的合并，解决冲突、删掉重复的，保证记忆库不乱；而且为了让用户觉得快，生成记忆得在智能体回复完用户后，在后台异步处理。只要跟踪好记忆的来源，做好“防记忆投毒”这类安全措施，开发者就能做出真正“懂用户、会成长”的智能助手。

# 尾注
1. https://cloud.google.com/use-cases/retrieval-augmented-generation?hl=en（检索增强生成相关文档）
2. https://arxiv.org/abs/2301.00234（上下文学习相关论文）
3. https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview（Agent Engine会话服务文档）
4. https://langchain-ai.github.io/langgraph/concepts/multi_agent/#message-passing-between-agents（多智能体消息传递相关文档）
5. https://google.github.io/adk-docs/agents/multi-agents/（ADK多智能体相关文档）
6. https://google.github.io/adk-docs/agents/multi-agents/#c-explicit-invocation-agenttool（“智能体即工具”调用相关文档）
7. https://agent2agent.info/docs/concepts/message/（智能体间通信协议相关文档）
8. https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/（智能体互操作性相关博客）
9. https://cloud.google.com/security-command-center/docs/model-armor-overview（Model Armor工具相关文档）
10. https://ai.google.dev/gemini-api/docs/long-context#long-context-limitations（Gemini长上下文限制相关文档）
11. https://huggingface.co/blog/Kseniase/memory（记忆相关博客）
12. https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory（语义记忆相关文档）
13. https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory（语义记忆相关文档，同12）
14. https://arxiv.org/pdf/2412.15266（原子事实相关论文）
15. https://arxiv.org/pdf/2412.15266（知识三元组相关论文，同14）
16. https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#sample-requests-text-gen-multimodal-prompt（多模态生成请求示例相关文档）
17. https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/generate-memories（Agent Engine记忆生成相关文档）
18. https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output（控制多模态生成结果相关文档）
19. https://cloud.google.com/agent-builder/agent-engine/memory-bank/set-up#memory-bank-config（Agent Engine记忆库配置相关文档）
20. https://arxiv.org/html/2504.19413v1（滚动总结相关论文）
21. https://google.github.io/adk-docs/tools/#how-agents-use-tools（ADK工具使用相关文档）
22. https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/generate-memories#consolidate-pre-extracted-memories（记忆整合相关文档）
23. https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/generate-memories#background-memory-generation（后台生成记忆相关文档）
24. https://arxiv.org/pdf/2503.08026（记忆重新排序相关论文）
25. https://google.github.io/adk-docs/callbacks/（ADK回调函数相关文档）
26. https://arxiv.org/html/2508.06433v2（过程性记忆相关论文）
27. https://cloud.google.com/blog/products/ai-machine-learning/rlhf-on-google-cloud（Google Cloud上的RLHF相关博客）
28. https://arxiv.org/pdf/2503.03704（记忆投毒相关论文）
29. https://cloud.google.com/security-command-center/docs/model-armor-overview（Model Armor工具相关文档，同9）
30. https://cloud.google.com/architecture/choose-design-pattern-agentic-ai-system（智能体系统设计模式选择相关文档）