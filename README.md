# RTX VSR — RunPod Serverless

既存のH3と分離した動画アップスケール専用Endpointです。
NVIDIA Video Effects SDK（nvidia-vfx 0.2.0.0）のVideoSuperResを使用し、処理後のMP4を返します。
ブラウザ再生時だけの高画質化ではありません。H3本体・LoRA・ComfyUIはインストールしません。

## 現在の検証範囲

- CPUテスト8件成功。入力制限、非対応形式/可変fpsの拒否、送受信・一時ファイル削除を確認。
- 実際のFFmpegを使用し、SDKだけを単純な画像拡大の偽物にしたテストで、フレーム数・fps・サイズ・音声引き継ぎを確認。
- 上記はNVIDIA SDKの画質・動作確認ではありません。
- CUDA13ベースイメージの存在、公式ComfyノードとSDK APIの呼び出し方、PyPI/NVIDIA配布方式を確認。
- 完全Dockerビルド、実GPUのSDK import/推論、画質・速度・ピークVRAMは**未検証**です。

## RunPod設定

1. Serverless → New Endpoint → GitHubでこのリポジトリ、main、/Dockerfile、Build context .、Queue。
2. 名前例：rtx-vsr-upscale。既存のH3 Endpointは変更しません。
3. GPUは対応するNVIDIA RTX GPU。最初の候補はRTX4090（実機検証前）。SDKはTensor Core搭載Turing以降のNVIDIA GPUを要求しますが、個別ホストは後述のsmoke_testで確認します。
4. Allowed CUDA versionsを13.0以降に設定。GPU1、Max workers1、Active workers0、Idle timeout5秒。
5. Container disk40GB、Initialization timeout900秒、Execution timeout900秒。開始コマンド・Model・追加ボリュームは不要。
6. Builds CompletedとReleasesの反映を確認。

CUDA13のPyTorchとSDKを使います。同梱PyTorchを引き継ぐvenvへ依存関係を入れるため、PEP 668のシステムPython上書きは行いません。
SDKはNVIDIAの配布インデックスからインストールされます。GitHubにSDKバイナリを再配布しません。
ビルド時にはパッケージの存在とCUDA版を確認し、GPUドライバが必要なSDK importはワーカーで行います。

## 最初の診断

```json
{"input":{"diagnose":true}}
```

GPU名、PyTorch/CUDA、SDK版を返します。ok=trueはimportとGPU認識の確認であり、推論成功ではありません。

次に、小さい合成グラデーション1枚を実際にVSRで処理します。

```json
{"input":{"smoke_test":true}}
```

ok=true、sdk_inference_verified=trueでSDK推論の成功を確認します。
このテストもGPUワーカーの料金対象です。こちらから自動送信しません。
失敗時にCPU拡大や別のモデルで代替せず、エラーとして返します。

## 動画の入力

```json
{
  "input": {
    "video_base64": "動画ファイルをbase64化した文字列",
    "scale": 2,
    "quality": "HIGH",
    "strength": 1.0
  }
}
```

- 2倍/3倍/4倍。出力の各辺はSDKに合わせ8px単位に丸めます。倍率によっては元の比率と数px差が生じます。
- 初期quality=HIGH。LOW/MEDIUM/HIGH/ULTRA/HIGHBITRATE_HIGH/HIGHBITRATE_ULTRAに対応。
- 動画の生成やプロンプト入力はありません。既存動画をVSRで処理します。品質や復元される細部は元動画に依存します。
- 入力6MiBまで、最大15秒、1〜60fps、2〜900フレーム、各辺64〜1920px、総面積1920×1080以下。
- 出力は各辺3840px以下・総面積3840×2160以下。SDR 8-bit・正方画素・固定fpsの通常動画が対象です。
- HDR・可変fps・参照URL・音声のみ・複数映像ストリームは未対応。入力はファイルのbase64だけです。
- 音声があれば先頭の音声トラックをAACへ再エンコードし、動画と一緒に返します。動画のfpsとフレーム数は維持します。
- H264/CRF20/veryfast、CPUエンコード。SDK拡大はGPU。現版はNVENCではありません。
- 応答のMP4は6MiBまで。超える場合はエラーになります。無断でビットレートを下げません。長尺・大容量は別途ストレージ返却方式が必要です。
- GPU SDKは別プロセスで実行し、10分を超えた処理はFFmpegごと停止します。一時入力・出力は成功/失敗後に削除します。
- フレームを1枚ずつ処理し、動画全体をGPUメモリに蓄積しません。

ローカルでJSONを作る場合（APIキー不要・送信なし）：

```sh
python make-request.py input.mp4 --scale 2 --quality HIGH --output request.json
```

動画アップロード用アプリとの接続は、SDKのsmoke_test確認後に行います。
H3アプリのEndpoint IDをこのEndpointへ変更すると入力契約が違うため動作しません。

## 結果

video_base64、mime_type=video/mp4、元サイズ/最終サイズ、フレーム数、fps、音声有無、SDK推論確認、処理時間、実行環境を返します。
worker_version=rtx-vsr-v1。

## 参照

- 公式Comfyノード：https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes
- NVIDIA Python SDK：https://pypi.org/project/nvidia-vfx/
- SDKガイド：https://docs.nvidia.com/maxine/vfx-python/latest/index.html

GPUでの動作や速度はsmoke_testと実際の短い動画で確認してください。ComfyHost掲載版と内部実装が同一とは主張しません。
