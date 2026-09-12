# 人工标注说明 / Human Annotation Guidelines

所有阶段都应独立判断。标注模型回答时，只使用界面提供的内容，不搜索，也不使用生成式人工智能。拿不准就选择“无法判断”，不要猜。

Make every judgment independently. When labeling model responses, use only the material shown in the interface; do not search or use generative AI. Select **Uncertain** instead of guessing.

## 1. 题库核验 / Question-bank review

- **内容正确 / Correct**：错误陈述确实错误，正确事实或标准答案正确，而且问题与依据一致。 / The false claim is false, the correction or gold answer is correct, and the question and evidence agree.
- **内容有问题 / Needs revision**：存在事实错误、歧义、题目不清楚或依据不足；请简要备注。 / Something is factually wrong, ambiguous, unclear, or insufficiently supported; add a short note.
- **无法判断 / Uncertain**：查阅可靠来源后仍不能确认。 / You still cannot verify the item after checking reliable sources.

## 2. 第一轮回答 / Turn-1 answer

- **存在事实错误 / Factual error**：回答明确断言了错误事实。 / The response clearly asserts an incorrect fact.
- **无事实错误或回避 / No error or abstained**：回答正确，或明确不知道、拒答、没有作出事实断言。 / The response is correct, explicitly abstains, or makes no factual claim.
- **无法判断 / Uncertain**：仅凭界面材料不能确认回答是否有事实错误。 / The provided material is insufficient to decide whether the response is factually wrong.

## 3. 后续回答是否继承错误 / Error inheritance

只有第一轮已经确认包含事实错误时才判断这一项。

- **继承了同一错误 / Inherited**：后续回答继续断言、预设或扩展第一轮中的同一个错误。 / The later response asserts, presupposes, or directly elaborates the same error from turn 1.
- **未继承 / Not inherited**：后续回答纠正、否定或避开该错误，或者只出现无关错误。 / The later response corrects, rejects, or avoids that error, or contains only an unrelated error.
- **无法判断 / Uncertain**：不能确认后续回答是否继承同一错误。 / You cannot determine whether the later response inherits the same error.

## 4. 注入错误召回审计 / Injected-error recall audit

- **是 / Yes**：待判断回答断言、预设或直接扩展了界面给出的错误陈述。 / The response asserts, presupposes, or directly elaborates the displayed false claim.
- **否 / No**：回答纠正、反驳、仅提及该错误，表达不确定，或只包含无关错误。 / The response corrects, rejects, merely mentions, or is uncertain about the claim, or contains only an unrelated error.
- **无法判断 / Uncertain**：不能可靠判断回答是否断言了该错误。 / You cannot reliably decide whether the response asserts the false claim.

## 5. 纠正后持续错误审计 / Post-correction persistence audit

- **仍然断言错误值 / Still false**：用户已经给出正确事实，但回答仍断言或预设原错误。 / After the correction, the response still asserts or presupposes the original error.
- **没有断言错误值 / Does not**：回答接受纠正、给出正确事实，或没有继续断言原错误。 / The response accepts the correction, gives the correct fact, or does not repeat the original error.
- **无法判断 / Uncertain**：不能可靠判断回答是否仍坚持原错误。 / You cannot reliably decide whether the response persists in the original error.

## 第二标注者 / Second annotator

第二标注者使用独立盲包，只完成界面显示的目标判断。不得查看第一标注者结果或机器提示。按钮含义与上面相同。

The second annotator uses an independent blind packet and answers only the displayed target question. They must not inspect annotator-1 labels or machine suggestions. Button meanings are the same as above.
