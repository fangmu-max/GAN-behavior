import pandas as pd
import torch
from generator import Generator
from dataset import BehaviorFeatureDataset, behavior_feature_collate_fn


def generate_behavior_feature(generator, input_topic, input_comment, input_created_at):
    """用生成器生成行为特征"""
    # 将输入的评论基本特征组成一个batch
    input_data = [(input_topic, input_comment, input_created_at)]
    input_dataset = BehaviorFeatureDataset(input_data)
    input_dataloader = torch.utils.data.DataLoader(
        input_dataset, batch_size=1, collate_fn=behavior_feature_collate_fn
    )

    # 生成行为特征
    with torch.no_grad():
        for batch in input_dataloader:
            batch = {k: v.to(generator.device) for k, v in batch.items()}
            behavior_feature = generator.generate_behavior_feature(batch)

    # 将行为特征从tensor转换为标量值
    behavior_feature = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in behavior_feature.items()
    }
    return behavior_feature


def main():
    # 读入输入数据
    input_df = pd.read_csv("input.csv")

    # 加载生成器
    generator = Generator()
    generator.load_state_dict(torch.load("best_generator.pth"))
    generator.eval()

    # 生成行为特征
    behavior_features = []
    for _, row in input_df.iterrows():
        behavior_feature = generate_behavior_feature(
            generator, row["topic"], row["comment"], row["created_at"]
        )
        behavior_feature["id"] = row["id"]
        behavior_features.append(behavior_feature)

    # 将行为特征保存到csv文件
    behavior_feature_df = pd.DataFrame.from_records(behavior_features)
    behavior_feature_df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()
