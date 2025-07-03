import timm 
import os

def print_timm_models(output_file="timm_models.txt"):
    """
    利用可能なtimmモデルの一覧をテキストファイルに出力します。
    
    Args:
        output_file (str): 出力先ファイルパス (デフォルト: "timm_models.txt")
    """
    try:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Available models in timm:\n")
            for model_name in timm.list_models():
                f.write(f"{model_name}\n")
        
        print(f"モデル一覧を {output_file} に保存しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # デフォルトのファイル名で実行
    print_timm_models()
    
    # 別のファイル名を指定することも可能
    # print_timm_models("output/timm_models_list.txt")