import re


class PIQA:
    #`__init__(self)`: 构造函数，用于初始化类的实例。
    # 在这个例子中，初始化了一个模板字符串 `_template`，用于构造问题。
    def __init__(self):
        self._template = "Question: {}\nAnswer:"
    #`get_context(self, examples)`: 用于获取示例的上下文信息。
    #接受一个`examples`字典作为参数，字典中包含了PIQA任务的示例数据。
    # 函数通过提取`examples`字典中的 `"goal"` 键对应的值作为上下文信息，
    # 并将其构造成一个字符串列表，每个字符串都是通过将上下文信息插入到模板字符串中得到的。
    def get_context(self, examples):
        ctx = examples['goal']
        return [self._template.format(c) for c in ctx]
    #`get_target(self, examples)`: 用于获取示例的目标答案。
    #接受一个`examples`字典作为参数，字典中包含了PIQA任务的示例数据。
    #函数首先判断示例数据中是否存在 `-1`，如果存在则表示这是测试集数据，返回一个空列表。
    #否则，根据示例数据中的 `"label"` 键和对应的值构造目标答案列表。
    #将每个标签与索引构成元组，并使用该元组来获取目标答案。
    def get_target(self, examples):
        if -1 in examples["label"]:  # test set
            return [""] * len(examples["label"])
        else:
            gt_tuples = [("sol{}".format(label + 1), idx)
                         for idx, label in enumerate(examples['label'])]
            return [examples[k][i] for k, i in gt_tuples]
#`PIQA`类提供了一个方便的方式来获取PIQA任务中的上下文信息和目标答案。这有助于在任务中对数据进行处理和准备，并为模型提供适当的输入和目标答案。

class HellaSwag:
    @classmethod
    #`preprocess(cls, text)`: 这是一个类方法（classmethod），用于对文本进行预处理。接受一个文本字符串作为参数。
    # 在预处理过程中，代码会去除文本两端的空白字符，替换"[title]"为句号，并使用正则表达式将方括号及其内部内容删除。
    # 接着将连续出现的两个空格替换为一个空格，并返回处理后的文本。
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
    #`get_context(self, examples)`: 用于获取示例的上下文信息。接受一个`examples`字典作为参数，字典中包含了HellaSwag任务的示例数据。函数首先使用 `zip` 拆分 `examples` 字典中的 `"activity_label"`、`"ctx_a"` 和 `"ctx_b"` 三个键对应的值。然后通过迭代操作将三个值拼接成一个字符串，并使用 `self.preprocess` 方法对其进行预处理。最终将预处理后的字符串构成的列表作为上下文信息返回。
    def get_context(self, examples):
        ctx_zip = zip(examples["activity_label"],
                      examples["ctx_a"], examples["ctx_b"])
        return [self.preprocess(a + ": " + b + " " + c.capitalize()) for a, b, c in ctx_zip]
    #`get_target(self, examples)`: 用于获取示例的目标答案。接受一个`examples`字典作为参数，字典中包含了HellaSwag任务的示例数据。函数从示例数据中获取 `"label"` 和 `"endings"` 键对应的值，然后通过迭代操作将每个标签对应的目标答案提取出来。如果标签为空字符串，则返回空字符串；否则，根据对应的标签值来获取 `endings` 列表中相应索引的答案，并使用 `self.preprocess` 方法对其进行预处理。最终将处理后的目标答案构成的列表返回。
    def get_target(self, examples):
        labels = examples["label"]
        endings = examples["endings"]
        targets = []
        for idx, label in enumerate(labels):
            target = '' if label == '' else endings[idx][int(label)]
            targets.append(self.preprocess(target))
        return targets


class OpenBookQA:
    def get_context(self, examples):
        return examples['question_stem']

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        targets = []
        for choice, answer in zip(choices, answers):
            answer = ord(answer.strip()) - ord('A')
            targets.append(choice['text'][answer])
        return targets


class ARC:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        for idx, answer in enumerate(answers):
            answer = num_to_letter.get(answer, answer)
            answer = ord(answer) - ord("A")
            answers[idx] = choices[idx]["text"][answer]
        return answers


class RACE:
    @classmethod
    def doc_to_text(cls, article, question):
        text = "Article: " + article + "\n\n"
        text += "Question: " + question + "\n\n"
        text += "Answer:"
        return text

    def get_context(self, examples):
        return [
            self.doc_to_text(article, question)
            for article, question in zip(examples["article"], examples["question"])
        ]

    def get_target(self, examples):
        answers = examples['answer']
        options = examples['options']
        for idx, answer in enumerate(answers):
            answers[idx] = options[idx][ord(answer) - ord("A")]
        return answers


class SciQ:
    def __init__(self):
        self._template = "{}\nQuestion: {}\nAnswer:"

    def get_context(self, examples):
        sources = examples['support']
        queries = examples['question']
        return [self._template.format(s, q) for s, q in zip(sources, queries)]

    def get_target(self, examples):
        return examples['correct_answer']


class WebQs:
    def get_context(self, examples):
        return ["Question: " + question + "\nAnswer:" for question in examples["question"]]

    def get_target(self, examples):
        return [" " + answers[0] for answers in examples["answers"]]



task_dict = {
    "piqa": PIQA(),
    "hellaswag": HellaSwag(),
    "openbookqa": OpenBookQA(),
    "arc_easy": ARC(),
    "arc_challenge": ARC(),
    "sciq": SciQ(),
    "web_questions": WebQs(),
    "race": RACE(),
}

#用于根据传入的参数映射数据集名称和配置名称。
def map_dataset_name_and_config(args):
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    if args.dataset_name == 'arc_easy':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Easy'
    elif args.dataset_name == 'arc_challenge':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Challenge'
    elif args.dataset_name == 'race':
        dataset_config_name = 'high'


    return dataset_name, dataset_config_name


LM_EVAL_TASK_NAME_MAPPING = {
    "web_questions": "webqs"
}
