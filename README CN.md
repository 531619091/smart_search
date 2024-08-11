# SmartSearch (智能搜索系统)

[English README](README.md)

SmartSearch 是一个先进的 AI 驱动搜索系统,能够将复杂的查询分解为可管理的子问题,进行信息搜索,并综合全面的答案。

## 功能特点

- 复杂查询分解
- 智能网络搜索
- 动态信息综合
- 多智能体协作

## 工作原理

1. 查询分解: Planner 智能体将主要查询分解为子问题。
2. 网络搜索: Search 智能体为每个子问题执行网络搜索。
3. 信息分析: Reflection 智能体分析搜索结果,并确定是否需要更多信息。
4. 答案综合: 一旦收集到足够的信息,系统将生成最终的全面答案。

## 安装

1. 克隆仓库:
git clone https://github.com/531619091/smart-search.git
2. 安装所需的包:
pip install -r requirements.txt

## 使用方法

运行主脚本:
python smart_search.py

## 配置

调整 `SmartSearch` 类中的 `max_turn` 参数可以控制最大搜索迭代次数。

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目使用 MIT 许可证 - 查看 LICENSE 文件了解详情。

## 示例查询

以下是一些示例查询,展示了 Smart Search 系统的功能:

1. "比较太阳能和风能作为可再生能源的优缺点"
2. "解释量子计算的基本原理及其潜在应用"
3. "分析全球气候变化对农业生产的影响"

## 系统流程

1. 用户输入复杂查询
2. Planner 智能体将查询分解为多个子问题
3. 对每个子问题:
   - Search 智能体执行网络搜索
   - Reflection 智能体分析搜索结果
   - 如果需要更多信息,重复搜索过程
4. 当收集到足够信息时,系统生成综合答案
5. 将最终答案呈现给用户

## 注意事项

- 确保您有足够的 API 使用额度,因为系统可能会进行多次 API 调用。
- 搜索结果的质量可能会影响最终答案的准确性。
- 对于非常复杂或专业的查询,可能需要多轮迭代才能得到满意的答案。

## 常见问题

Q: 系统支持哪些语言？
A: 目前系统主要支持中文和英文查询。

Q: 如何调整搜索的深度和广度？
A: 您可以通过修改 `smart_search.py` 中的相关参数来调整搜索行为。

Q: 系统是否会保存搜索历史？
A: 目前系统不会保存搜索历史。每次查询都是独立的会话。

如果您在使用过程中遇到任何问题或有任何建议,请随时提出 issue 或联系我们。