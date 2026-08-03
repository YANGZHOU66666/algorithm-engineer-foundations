# Learn Claude Code

harness engineering：除了LLM本身都是harness。要么调整LLM参数（训练），要么构建harness

模型做决策。Harness 执行。模型做推理。Harness 提供上下文。模型是驾驶者。Harness 是载具。

Harness = Tools + Knowledge + Observation + Action Interfaces + Permissions

    Tools:          文件读写、Shell、网络、数据库、浏览器
    Knowledge:      产品文档、领域资料、API 规范、风格指南
    Observation:    git diff、错误日志、浏览器状态、传感器数据
    Action:         CLI 命令、API 调用、UI 交互
    Permissions:    沙箱隔离、审批流程、信任边界

Harness工程师的职责：

- **实现工具。** 给 agent 一双手。文件读写、Shell 执行、API 调用、浏览器控制、数据库查询。每个工具都是 agent 在环境中可以采取的一个行动。设计它们时要原子化、可组合、描述清晰。
- **策划知识。** 给 agent 领域专长。产品文档、架构决策记录、风格指南、合规要求。按需加载（s05），不要前置塞入。Agent 应该知道有什么可用，然后自己拉取所需。
- **管理上下文。** 给 agent 干净的记忆。子 agent 隔离（s04）防止噪声泄露。上下文压缩（s06）防止历史淹没。任务系统（s07）让目标持久化到单次对话之外。
- **控制权限。** 给 agent 边界。沙箱化文件访问。对破坏性操作要求审批。在 agent 和外部系统之间实施信任边界。这是安全工程与 harness 工程的交汇点。
- **收集任务过程数据。** Agent 在你的 harness 中执行的每一条行动序列都是训练信号。真实部署中的感知-推理-行动轨迹是微调下一代 agent 模型的原材料。你的 harness 不仅服务于 agent -- 它还可以帮助进化 agent。

## 核心闭环

### s01 agent loop

“真正的 agent 起点，是把真实工具结果重新喂回模型，而不只是输出一段文本。“

最基础的智能体，每次LLM根据上文选择调用工具还是输出最终答案。如果调用工具，获得工具结果并开启下一轮，否则结束。

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until stop_reason != "tool_use")
```

 伪代码：

```
agent_loop(){
	messages = [用户输入]
    while(true){
        llm回复 = 调用LLM(messages)
        messages.push_back(llm回复)
        if(llm回复=某工具调用){
            工具回复 = 调用该工具()
            messages.push_back(工具回复)
        }else{
            break
        }
    }
}
```

### s02 tool use

“主循环本身不用变复杂；工具能力靠一层清晰的路由面增长。”

为工具调用包装一层，解决纯bash工具调用可能做出不安全行为的问题

加两个组件：

1. dispatch_map，将模型实际输出的工具名映射为具体的可执行函数
2. safe_path(p: str)，判断p是否为安全路径，是返回p，不是报错



### s03 todo write

“对多步骤任务来说，可见计划不是装饰，而是防止会话漂移的稳定器。”

问题：虽然最开始可能定了计划，但由于上下文越来越长，如果没有一块**显式、稳定、可反复更新**的计划状态，大任务就很容易漂

解决：给主Agent一个todo工具，让agent把当前会话里的计划外显出来，并且持续更新。总体思路：主Agent定计划，全部未完成->按顺序做，完成一项就打一个完成；如果好几轮没更新，就提醒一下->直到所有完成，结束任务

具体实现：

【单个任务条目】拥有任务内容content、任务状态status（pending、in_progress、completed三个枚举值）、进行中时更自然的进行时描述activeForm

【运行状态】任务列表+提醒轮次数（多少轮过去后模型必须更新计划）

【状态约束】同一时间最多一个in_progress

把更新或创建todo list的方法做成LLM的工具，调用时传入每个任务的描述和当前状态，返回内容为当前任务列表的字符串拼接

一个例子：

- 完成任务（用TASK代替），需要三个子任务（用subtask1、subtask2、subtask3代替），调用工具有TODO（创建或更新任务列表）、其他tool（用tool1、tool2、tool3等代替）

```
messages列表/任务列表变化的过程：

1. 用户输入任务描述
messages = [
	{user: TASK}
]

2. 大模型返回任务列表，提醒轮次置0
messages = [
    {user: TASK},
    {assistant:use tool TODO}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

3. 系统调用工具，返回工具调用结果，提醒轮次不到3，不提醒
messages = [
    {user: TASK},
    {assistant:use tool TODO},
    {user: tool TODO result}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

4. 模型为完成subtask1，尝试调用工具tool1
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

5. 系统调用tool1，返回工具调用结果，提醒轮次+1=1，但不到3，不提醒
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1},
    {user tool1 result}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

6. 模型调用tool2，但还没有调用TODO
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1},
    {user: tool1 result},
    {assistant: use tool2}
]

task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

7. 系统调用tool2，返回工具调用结果，提醒轮次+1=2，但不到3，不提醒
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1},
    {user: tool1 result},
    {assistant: use tool2},
    {user: tool2 result}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

8. 模型调用tool3+tool4，依然没有调用TODO
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1},
    {user: tool1 result},
    {assistant: use tool2},
    {user: tool2 result},
    {assistant: use tool3, tool4}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]

9. 系统调用 tool3/tool4，返回工具调用结果；由于连续 3 轮未更新 TODO，触发 reminder
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: [tool TODO result]},
    {assistant: use tool1},
    {user: [tool1 result]},
    {assistant: use tool2},
    {user: [tool2 result]},
    {assistant: use tool3, tool4},
    {user: [reminder, tool3 result, tool4 result]}
]
task_list = [
    {subtask1: in_progress},
    {subtask2: pending},
    {subtask3: pending}
]

10. 模型调用TODO，假设这时候实际上subtask1已完成，因此任务列表改成subtask1 completed，subtask2 pending
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1},
    {user: tool1 result},
    {assistant: use tool2},
    {user: tool2 result},
    {assistant: use tool3, tool4},
    {user: [reminder, tool3 result, tool4 result]}
    {assistant: use tool TODO}
]
task_list = [
	{subtask1: in_progress},
	{subtask2: pending},
	{subtask3: pending}
]（这时候还没更新）

11. 系统调用TODO工具，task_list随之更新
messages = [
    {user: TASK},
    {assistant: use tool TODO},
    {user: tool TODO result},
    {assistant: use tool1},
    {user: tool1 result},
    {assistant: use tool2},
    {user: tool2 result},
    {assistant: use tool3, tool4},
    {user: [reminder, tool3 result, tool4 result]}
    {assistant: use tool TODO},
    {user: tool TODO result}
]
task_list = [
	{subtask1: completed},
	{subtask2: in_progress},
	{subtask3: pending},
]

后续同理
```

### s04 subagent

“把探索性工作移进干净上下文后，父 agent 才能持续盯住主目标。”

问题：如果让主Agent执行一些局部任务，中间过程上下文太长，不能保留真正有用的东西

解决：做局部任务让主Agent分发给subagent解决

具体思路：给主Agent一个task工具，传参有prompt字段，用于子Agent的第一条user prompt。子Agent有一套较小的工具集，message与主Agent隔离，最终工具调用的返回结果为子Agent最后一条response（即不带工具调用的）。注意：需要定义子Agent的退出条件，如完成任务、设置最大轮数

### s05 skill

“专门知识不该一开始全部塞进上下文，而该在需要时被轻量发现、按需展开。”

问题：完成一个任务，干不同事需要的知识不一样，全放进system prompt会耗费token、注意力涣散

解决：system prompt中只加载所有skill的metadata，需要时加载全skill

具体思路：在system prompt中写所有skill的描述，形如

```
Skills available:
- skill_name_1: description of skill 1
- skill_name_2: description of skill 2
```

再添加一个load_skill的工具，入参为skill name，返回值为skill的全部内容，放到最新一条user prompt（工具调用结果）里



### s06 context_compact

“压缩的目标不是删历史，而是保住连续性和下一步所需的工作记忆。”

问题：长上下文太长，浪费token/关注不到重点；上下文超过最大长度。

解决：多级压缩机制。1. 超过2000字符的工具调用结果只展示2000字，其余存硬盘；2. 超过3轮的旧工具调用结果，使用占位符代替；3.  上下文总长度达到上限时，用LLM概括一下作为所有历史

具体实现：

1. 读一个超长文件（假如30000字符）。实际返回值加到messages里为：

  ```
  [{ 
      "role": "user", 
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "call_1",
          "content": "<persisted-output>\nFull output saved to: .task_outputs/tool-results/call_1.txt\nPreview:\n[这里只有前2000个字符...]\n</persisted-output>"
        }
      ]
    }]
  ```

2. 当工具调用了几轮之后，假如已经有4个工具调用的结果了，最旧的工具调用结果被占位符取代：

   ```
   [
     ...
     { "role": "assistant", "content": [ { "type": "tool_use", "id": "call_1", ... } ] },
     { 
       "role": "user", 
       "content": [
         {
           "type": "tool_result",
           "tool_use_id": "call_1",
           "content": "[Earlier tool result compacted. Re-run the tool if you need full detail.]" 
         }
       ]
     },
     { "role": "user", "content": [ call_2 的完整结果 ] },
     { "role": "user", "content": [ call_3 的完整结果 ] },
     { "role": "user", "content": [ call_4 的完整结果 ] }
   ]
   ```

3. 假如即将超出上下文长度了，messages彻底重置，变成一条概括的user prompt：

   ```
   [
     { 
       "role": "user", 
       "content": "This conversation was compacted so the agent can continue working.\n\n[Summary]:\n1. 目标：修复 src/app.py 内存泄漏。\n2. 进展：已分析 server.log，发现泄漏点在 App.listen()；已通过 edit_file 修复。\n3. 状态：修复已完成，正在验证测试结果。\n4. 最近文件：src/app.py, logs/server.log" 
     }
   ]
   ```

## 系统加固

### s07 permission system

“模型产生的执行意图，必须先通过清晰的权限门，再变成真正动作。”

问题：模型可能写错文件，执行危险命令，在不该干的地方执行tool

解决：在不同模式下，给不同类型的tool以及不同路径加上不同权限；一部分权限是模式固有的，一部分是随着用户手动的允许与禁止，动态增删的

具体实现：

规则长什么样？tool_name + content + behavior(deny, ask, allow)，说明某工具若含某内容则应当被怎么处理

```json
{
    "tool": "bash",
    "content": "sudo *",
    "behavior": "deny",
}
```

0~4共5个step：

0. 特殊处理bash中的极危险操作（sudo，rm -rf等），直接deny
1. 遍历处理所有deny的规则
2. 根据mode不同，deny/allow特定类型的规则（如plan模式deny所有write的而allow所有read的）
3. 遍历处理所有allow的规则
4. 剩下的ask用户

deny在mode前的原因：防止特定危险操作被放行（如读到不该读的地方）

### s08 hook

“Hook 让系统围绕主循环生长，而不是不断重写主循环本身。”

三种hook：SessionStart、PreToolUse、PostToolUse

原理：

```
r = subprocess.run(...)
```

根据r的情况决定将信息注入message或拦截工具调用（只针对PreToolUse Hook）

对SessionStart Hook：在对话最开始时自动执行，输出内容不放进messages，只在控制台里输出一下

对PreToolUse Hook：模型给出tool建议、系统执行tool函数前执行。可以根据hook调用情况决定要不要执行tool；可以将hook的信息放到messages里（工具调用结果的前面）。例如，可以在修改配置文件的tool执行之前在message里加一个"[Hook message]: 注意：你正在修改配置文件"

对PostToolUse Hook：系统执行tool函数后执行。可以将hook的信息放到messages里，例如在tool执行之后在messages里加一个"建议先运行最小测试"



### s09 memory

“只有跨会话、无法从当前工作重新推导的知识，才值得进入 memory。”

问题：重复忘记一些事情，如用户长期偏好/纠正过的错误/项目约定等

关键点：memory不是什么都存（可能过时/越存越乱）；跨会话仍有价值、不能轻易从当前仓库推导出来的信息，适合存

具体方法：

- 有关memory本身：

1. 定义几类memory（有4种type：user-用户偏好、feedback-用户明确纠正过的地方、project-**不容易从代码直接重新看出来**的项目约定或背景、reference-外部资源指针）

2. 准备save_memory工具，参数：name、description、type、content；每条都存成一个md文件。文件内容形如：

   ```markdown
   ---
   name: prefer_tabs
   description: 用户偏好使用 Tab 缩进而非空格
   type: user
   ---
   用户在多次会话中明确要求，在编辑本项目源码时必须使用 Tab 进行缩进。
   即使是在 Python 这种默认使用空格的语言中，也要遵循这一约定。
   违反此规定会导致用户提出负面反馈。
   ```

   同时，会有记忆的索引区，维护所有记忆的目录：

   ```markdown
   # Memory Index
   
   - prefer_tabs: 用户偏好使用 Tab 缩进而非空格 [user]
   - mock_tests_feedback: 用户反对在测试中使用过重的 Mock [feedback]
   - internal_api_doc: 内部 API 接口文档地址 [reference]
   ```

3. 每次会话时，将所有memory取出来拼进system prompt

- DreamConsolidator：

dream机制，将多个记忆总结为较少条目的记忆，有触发条件，如24h只能触发一次、至少有5条记忆才能触发等

- 补充说明：

记忆需要分private/team权限

### s10 system prompt

”模型看到的不是一坨固定 prompt，而是一条按阶段拼装的输入流水线。“

问题：如果system prompt写死，不容易维护/测试/动态更新

解决：将多个来源分别加载，拼成最终system prompt

来源：

- _build_core()：包含 Agent 的基本身份声明（如操作目录）和最高行为准则（如“先读后猜”、“验证假设”）。

- _build_tool_listing()：列出所有可用工具的名称、参数 schema 和功能描述，确保模型了解如何调用外部能力。

- _build_skill_listing()：扫描 skills/ 目录下的 SKILL.md 文件，提取技能的名称和描述，为模型提供高层级的能力概览。

- _build_memory_section()：从 .memory/ 目录加载持久化记忆（排除索引文件 MEMORY.md ）。格式通常为 [类型] 名称: 描述 \n 内容正文 ，用于注入长期沉淀的项目背景或用户偏好。

- _build_claude_md()：按照优先级（用户全局 > 项目根目录 > 当前子目录）聚合所有 CLAUDE.md 文件中的具体指导规范。

- 动态边界符：插入常量 DYNAMIC_BOUNDARY (即 \=== DYNAMIC_BOUNDARY \=== )

  核心设计意图：将相对稳定的静态指令（上述 1-5 项）与高频变化的动态信息分隔开，便于在实际工程中对静态部分进行缓存以节省 Token。

- _build_dynamic_context()：包含实时信息，如当前日期、工作目录路径、模型 ID 和运行平台。

system prompt示例：

```markdown
You are a coding agent operating in k:\大四\learn-claude-code.
Use the provided tools to explore, read, write, and edit files.
Always verify before assuming. Prefer reading files over guessing.

# Available tools
- bash(command): Run shell commands in the workspace
- read_file(path, limit): Read content from a file
- write_file(path, content): Create or overwrite a file
- edit_file(path, old_text, new_text): Replace a specific block of text in a file

# Available skills
- agent-builder: Design and build AI agents for any domain. Use when users ask to "create an agent" or "build an assistant".
- code-review: Automated code analysis and improvement suggestions.

# Memories (persistent)

[project] coding-style: coding-style
Always use type hints for Python functions. Prefer async/await for I/O bound tasks.

[user] preference: preference
User prefers Chinese for explanations and English for code comments.

# CLAUDE.md instructions

## From project root (CLAUDE.md)
- Test command: `pytest tests/`
- Lint command: `ruff check .`
- Follow PEP 8 style guide.

=== DYNAMIC_BOUNDARY ===

# Dynamic context
Current date: 2026-04-17
Working directory: k:\大四\learn-claude-code
Model: claude-3-5-sonnet-20240620
Platform: Windows
```



### s11 error recovery

”系统必须清楚自己此刻是在继续、重试，还是处于恢复流程。“

主要解决的是模型调用本身出的一些问题，和Agent系统关系不太大

三个实现：

1. 输出被截断时，做续写：当失败原因为超过max_tokens时，再给一条固定的prompt让继续输出，直到输出完或达到最大截断次数
2. 上下文太长时，先压缩再重试：调用context_compact工具
3. 连续抖动时，退避重试：当因为网络超时、连接错误或 API 频率限制，间隔指数级增长地重试。使用公式 $base * 2^{attempt} + jitter$ 计算延迟时间，jitter是随机数，防止大批量API调用在同一时刻重试

## 任务运行时

### s12 task system

“Todo 适合会话内规划，持久任务图才负责跨步骤、跨阶段协调工作。”

和03 todo的核心区别：todo更多强调一个会话内，完成一个任务的多个步骤，todo list不会放在持久存储中而是在内存中；task system则是强调更大粒度的“任务”，以及任务之间的依赖关系（即，某任务必须在哪几个任务执行完之后才能执行），后续也会为多智能体协作打下基础，任务图存在硬盘中，是持久化的。

实现：

单个task存成一条json文件，说明上下游依赖关系、当前状态、任务说明；

给智能体4个工具，1. 创建任务；2. 更新任务状态（包括上下游依赖关系的更新）；3. 查看task_list（只会给每个task的简短说明、状态和上下游依赖关系，不会说任务细节）；4. 查看单条task的detail

伪代码例子：

```
- 用户： 查一下 xxx，然后把内容写到 a 文件里
- 模型： task_create("查一下 xxx") ， task_create("把查询结果写到 a 文件里")
- 工具结果：任务 1、任务 2 创建成功
- 模型： task_update(task_id=1, addBlocks=[2])
- 工具结果：任务 1 现在会阻塞任务 2；任务 2 的 blockedBy 自动补上 1
- 模型： task_list()
- 工具结果：任务 1 未完成；任务 2 被任务 1 阻塞
- 模型： task_update(task_id=1, status="in_progress") + 若干查询工具
- 工具结果：任务 1 进入进行中；查询结果返回
- 模型： task_update(task_id=1, status="completed")
- 工具结果：任务 1 完成；系统自动把任务 1 从任务 2 的 blockedBy 中移除
- 模型： task_list()
- 工具结果：任务 2 未完成，但现在已可执行
- 模型： task_update(task_id=2, status="in_progress") + write_file("a", ...)
- 工具结果：任务 2 进入进行中；文件写入成功
- 模型： task_update(task_id=2, status="completed")
- 工具结果：任务 2 完成
- 模型：输出“我完成了所有任务……”

至此，循环结束
```

### s13 background task

“持久任务描述要完成什么，运行槽位描述谁在跑、跑到哪里；两者相关但不是一回事。”

最小心智模型：

```
主循环
  |
  +-- background_run("pytest")
  |      -> 立刻返回 task_id
  |
  +-- 继续别的工作
  |
  +-- 下一轮模型调用前
         -> drain_notifications()
         -> 把摘要注入 messages

后台执行线
  |
  +-- 真正执行 pytest
  +-- 完成后写入通知队列
```

维护两个数据结构，后台任务列表+通知队列。当LLM调用了需要后台执行的tool时（通过封装一个额外的工具background_run来实现），则添加一个后台任务（状态为运行中）的记录，在user prompt里说明xx后台任务开始进行，不阻塞Agent Loop的运行；等到后台任务执行完毕，将后台任务记录更新为completed，并添加结果摘要和完整结果地址，存完整的运行结果到一个文件中；放一个完成的通知进入通知队列中。下一轮LLM请求前，将通知队列中的所有内容作为user prompt放进messages里，再进行请求。

### s14 cron scheduler

“当任务能后台运行以后，时间本身也会变成另一种启动入口。”

（个人感觉这章实现的功能不是特别合理）

有关模块：时间调度器，消息队列

后台维护一个基于时间的任务调度器，每到对应的时间点，放一条预设好的消息到消息队列里。每当调用LLM之前，把消息队列里的消息放进messages里作为user prompt，以做到定时提醒的效果



## 多agent平台

### s15 agent teams

“系统一旦长期运行，就需要有名字、有身份、可持续存在的队友，而不只是一次性子任务。”

teammate：持久的，有生命周期的，有名字、角色、消息入口的Agent

具体实现：.team文件夹下维护一个config.json，里面存有team_name和所有team_members的配置。每个team_member有几个属性：name、role（创建时设置好）、status（working/idle/shutdown）

给lead agent一个spawn工具，可以创建新的teammate/将idle的teammate激活为working，并传进去初始任务让其开始执行

lead agent和所有teammate agent都有send_message工具，可以向对应agent发送“邮件”。“邮件”写在./team/inbox/{member_name}.jsonl中，所有agent（包括teammate和lead）在agent loop一轮的开始强制读一次对应的“邮箱”，若有内容则读取，拼接到messages里，成功读取后会去掉“邮箱”里的邮件。teammate在认为合适的时候（比如完成了某项任务），会调用send_message给lead发“邮件”。

当teammate agent完成任务（即有一轮没有调用工具）或达到loop上限，从working变成idle。不会主动激活，必须要lead spawn才会。idle状态下，也不会读邮件（因为就没有agent loop），邮件一直持久化在文件里。注意，teammate agent的messages不是持久的，idel之后重新spawn会从空数组开始。

### s16 team protocols

“团队只有在”协作遵守共同消息模式时，才会变得可理解、可调试、可扩展。“

本章核心：封装了两种request，1. plan申请工具[plan_approval]（teammate向lead提交plan，lead审批），2. shutdown请求工具[shutdown_request]（lead让teammate关机，teammate给回应，使用shutdown_response工具）；request结构会持久化在硬盘里。

此外，邮箱机制依然存在，teammate和lead之间仍可以通过inbox进行通讯。只是额外封装了两种请求。

request的结构体形如：

```
### 通用层
所有 request 都有：
- 标识： request_id
- 类型： kind
- 参与者： from 、 to
- 状态： status
- 时间： created_at 、 updated_at

### 业务层
不同 request 再加自己的字段：
- shutdown：
  - resolved_by
  - resolved_at
  - response
- plan approval：
  - plan
  - reviewed_by
  - resolved_at
  - feedback
```

请求过程：

情形一：某teammate在运行过程中认为某事情拿不准，先写了个计划给lead审批，调用plan_approval工具，在硬盘中创建了一个request结构体（状态pending）；lead邮箱中收到有新plan request，因此这轮调用执行plan_approval工具进行审批（approve/reject）。审批结果发到teammate的邮箱里。

情形二：lead在运行过程中认为某teammate已经不需要继续运行了，调用shutdown_request工具，在硬盘中创建一个类型为shutdown的request结构体（状态pending）；该teammate邮箱中收到新的shutdown request，因此这轮调用执行shutdown_response工具进行确认，如果 approve 关闭请求，随后退出 loop，并把状态写成 shutdown ”；如果 reject，就不会关机。确认结果发到lead的邮箱里。



### s17 autonomous agents

”自主性开始于：队友能安全找到可做的事、认领它，并带着正确身份继续执行。“

前面一些章节的大综合。维护一个.task文件夹，里面存着所有要完成的任务；维护一个team，有inbox和requests

当每个teammate做完手头的事情之后，变成idle状态，但idle状态会定时触发轮询，看是否有角色匹配的任务/对应inbox是否有邮件。如果有，teammate变成working状态。如果长时间没活，自己变成shutdown状态



### s18 worktree task isolation

”task 管目标，worktree 管隔离执行车道和收尾状态；两者不能混成一个概念。“

创建task之后，再为这个task分一个独立的工位worktree，然后所有和这个任务相关的工作都尽量在这个文件夹下进行。还要尽可能记录下日志

任务完成后，再确认一下这个工位是需要删除还是保留。没有必要保留的话（比如中间过程和产物文件不重要），就可以删除



### s19 mcp plugin

”外部能力系统不该是外挂；它们应和原生工具一起处在同一控制面上。“

把外部MCP接入的工具，转化为和系统内工具完全一致的格式。

包括工具列表内的形式、权限检验（在assistant给要执行的工具和参数后，进行check）、路由思路、结果格式。
