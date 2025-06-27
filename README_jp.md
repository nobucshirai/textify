# textify

[![Version](https://img.shields.io/github/v/release/nobucshirai/textify?include_prereleases)](https://github.com/nobucshirai/textify/releases)
[![Tests](https://img.shields.io/github/actions/workflow/status/nobucshirai/textify/pytest.yml?label=pytest)](https://github.com/nobucshirai/textify/actions)
[![License](https://img.shields.io/github/license/nobucshirai/textify)](https://github.com/nobucshirai/textify/blob/main/LICENSE)

音声・動画・文書ファイルを一括処理するモジュラー型Pythonパッケージです。音声・動画ファイルには[Whisper](https://github.com/openai/whisper)、PDF・画像ファイルには[EasyOCR](https://github.com/JaidedAI/EasyOCR)を使用します。異なる処理タイプに特化したモジュールを持つクリーンなモジュラーアーキテクチャを特徴とし、CPU/GPU 使用率、消費電力、処理時間、システム情報を包括的にログ出力して、ワークフローの最適化に役立てます。

## 特徴

* **モジュラーアーキテクチャ**：メディア処理、文書処理、システム監視、ユーティリティ用の専用モジュールによる責務の明確な分離
* **一括処理**：引数として渡された個別ファイルを処理するか、指定したディレクトリ内の新しいファイルを自動的に検出して処理します
* **マルチフォーマット対応**  
  * Whisper による音声・動画の文字起こし  
  * EasyOCR による PDF／画像のテキスト抽出  
    * PDF は直接テキストを取り出し，無い場合のみ OCR にフォールバック
* **リソース監視**：CPU/GPU 利用率，消費電力 **および GPU の総エネルギー消費量 (Wh)** を記録します
* **デバイス選択**：GPU（`cuda`）または CPU を自動的に切り替え、CUDA 利用不可時には CPU にフォールバックします
* **処理時間推定**：対応する NVIDIA GPU（RTX 4070、4060 Ti）向けに処理時間を推定します
* **詳細なログ**：タイムスタンプ付きコンソール & ファイルログ（1 つのログファイルに追記）。各処理ごとにダンプファイルを生成します

## クイックスタート

簡単なインストールと基本的な使い方:

```bash
# 仮想環境の作成と有効化
python -m venv textify-env
source textify-env/bin/activate

# textify のインストール
pip install git+https://github.com/nobucshirai/textify.git

# 特定のファイルを処理
textify audio1.mp3 video1.mp4 document.pdf image.jpg

# ディレクトリを指定して実行 (デフォルトでGPUを使用)
textify --input-dir /path/to/media_dir

# CPU のみを使用して英語の文字起こしを行う場合
textify --input-dir /path/to/media_dir --device cpu --language English
```

より高度なオプションと設定については、以下の詳細セクションをご覧ください。

## 前提条件

* Ubuntu 24.04 LTS で動作確認済み
* Python 3.8 以上
* `openai-whisper` Python パッケージ（pip 経由でインストール）
* CPU 監視用 `psutil`（任意推奨）
* NVIDIA GPU 監視用 `nvidia-ml-py`（pynvml）（任意）
* メディア再生時間推定用 `ffprobe`（FFmpeg から）（任意）
* GPU サポートに必要な NVIDIA ドライバおよび NVML ライブラリ（`--device cuda` 使用時）
* 文書・画像処理用 `easyocr`（任意）
* PDF処理用 `PyMuPDF`（任意）

## パッケージ構造

textifyパッケージは、より良い保守性と拡張性のためにモジュラーアーキテクチャを採用しています：

```text
textify/
├── __init__.py         # パッケージ初期化とエクスポート
├── core.py             # コア処理ロジックとメインエントリーポイント
├── cli.py              # コマンドライン インターフェースと引数解析
├── utils.py            # ユーティリティ関数、フォーマット、ファイル管理
├── system.py           # システムチェック、監視、ハードウェア検出
├── media.py            # Whisperによる音声・動画処理
└── documents.py        # OCRによる文書・画像処理
```

### モジュール責務

* **`core.py`**: メインアプリケーションワークフロー、ログ設定、プロセス調整
* **`cli.py`**: コマンドライン引数解析と検証
* **`utils.py`**: 時間フォーマット、ファイル発見、分類、ログユーティリティ
* **`system.py`**: ハードウェア検出（CUDA、GPU、ffprobe）、リソース監視
* **`media.py`**: 音声・動画時間抽出、Whisperモデル読み込み、文字起こし
* **`documents.py`**: EasyOCRとPyMuPDFによるPDFと画像処理

## インストール

まず、仮想環境を作成して有効化します：

```bash
# 仮想環境を作成
python -m venv textify-env

# 仮想環境を有効化
source textify-env/bin/activate
```

GitHub からインストール：

```bash
pip install git+https://github.com/nobucshirai/textify.git
```

## 使い方

textifyパッケージはインストール後、複数の方法で使用できます：

1. **パッケージコマンド**: インストール済みの`textify`コマンドを使用
2. **Python インポート**: コア機能をプログラムでインポートして使用

### パッケージコマンド（推奨）

pip でインストール後、`textify` コマンドを使用します：

パッケージは2つの動作モードをサポートします：

1. **ファイルモード**：引数として渡された特定のファイルを処理
2. **ディレクトリモード**：`--input-dir` を使用してディレクトリ内の未処理ファイルをすべて処理

```bash
# ディレクトリモード
textify \
  --input-dir ./media \
  --log-file ./logs \
  --model medium \
  --device cpu

# ファイルモード
textify \
  audio1.mp3 video1.mp4 document.pdf \
  --model medium \
  --language English \
  --device cpu
```

### プログラムでの使用

独自のPythonスクリプトでコア機能をインポートして使用できます：

```python
from textify.core import main

# カスタム引数で実行
import sys
sys.argv = ['textify', '--input-dir', '/path/to/media', '--device', 'cpu']
main()

# または個別モジュールを使用
from textify.media import process_audio_video_files
from textify.documents import process_document_files
from textify.system import initialize_system_checks
```

### コマンドライン引数

* `files`（位置引数）：処理するメディアファイル（`--input-dir` と同時使用不可）
* `--input-dir`：処理対象のメディアファイルが格納されたディレクトリ（ファイル引数と同時使用不可）
* `--log-file`：ログ出力先のパスまたはディレクトリ。ディレクトリ指定時は日付入りのログファイルが作成されます
* `--monitoring-interval`：リソース使用率サンプリング間隔（秒）（デフォルト：`10`）
* `--gpu-threshold`：処理を許可する GPU 利用率の閾値（%）（デフォルト：`20`）
* `--ignore-gpu-threshold`：現在の GPU 利用率に関係なく処理を行います
* `--model`：Whisper モデル名（デフォルト：`large`）
* `--language`：文字起こし言語（デフォルト：`Japanese`）
* `--device`：処理デバイス（`cuda` または `cpu`）（デフォルト：`cuda`。CUDA が利用できない場合は自動的に `cpu` を使用）
* `-w`, `--watch`：`--input-dir` を監視し、新規ファイルを検出次第処理します（watchdog 使用）。

**注意**：`--input-dir` またはファイル引数のいずれかを指定する必要がありますが、両方を同時に指定することはできません。

## 対応ファイル形式

* **音声**：mp3、wav、aac、flac、ogg、m4a、wma
* **動画**：mp4、mov、avi、wmv、flv、mkv、webm、m4v、mpg、mpeg、他多数
* **文書**：pdf
* **画像**：jpg、jpeg、png、bmp、tiff、tif、webp、gif、heic、heif

## ログ

* **ログファイル**：システム情報、リソース使用率、推定および実際の処理時間など、タイムスタンプ付きエントリを含みます
* **ダンプファイル**：処理された各ファイルと同じディレクトリに `_dump.txt` ファイルを生成し、以下を含みます：

  * 開始 & 終了タイムスタンプ
  * メディア再生時間（音声・動画ファイルの場合）
  * 推定 & 実際の処理時間
  * WhisperまたはOCRの出力全体

## 出力ファイル

処理された各ファイルに対して、以下のファイルが作成されます：

* テキスト抽出ファイル：`ファイル名_拡張子.txt`の形式（例：`lecture_mp3.txt`、`document_pdf.txt`）
* ダンプファイル：`ファイル名_拡張子_dump.txt`の形式で、処理の詳細情報と完全な出力を含む

## 使用例

* **GPU で日本語文字起こし（ディレクトリモード）**：

  ```bash
  textify --input-dir ./media
  ```

* **特定ファイルを CPU で英語文字起こし**：

  ```bash
  textify \
    audio1.mp3 video1.mp4 document.pdf \
    --language English \
    --device cpu
  ```

* **カスタムログディレクトリを指定する場合（ディレクトリモード）**：

  ```bash
  textify \
    --input-dir ./media \
    --log-file ./logs
  ```

* **異なるディレクトリのファイルを処理**：

  ```bash
  textify \
    /path/to/audio1.mp3 \
    /another/path/video1.mp4 \
    ./local/document.pdf
  ```

## systemd による自動処理

textify にはディレクトリを監視する `--watch` オプションがあります。`watchdog` ライブラリを用いて新規ファイルを検出し、systemd と組み合わせれば常駐処理も可能です。

単に監視したいだけなら次のように実行します。

```bash
textify --input-dir /path/to/media --watch
```

### 設定手順

1. `~/.config/systemd/user/` に `watch_textify.service` などのサービスファイルを作成します。

```ini
[Unit]
Description=Watch Textify Directory
After=network.target

[Service]
Type=simple
ExecStart=/path/to/venv/bin/textify \
    --input-dir /path/to/media \
    --watch \
    --log-file /path/to/logs
Restart=always
Environment=PATH=/path/to/textify-env/bin:/usr/bin:/bin
WorkingDirectory=/path/to/working/directory

[Install]
WantedBy=default.target
```

2. サービスを有効化して起動します。

```bash
systemctl --user daemon-reload
systemctl --user enable watch_textify.service
systemctl --user start watch_textify.service
systemctl --user status watch_textify.service
```

この設定により、指定したディレクトリを継続的に監視し、新しいファイルを検出次第自動で処理します。


## Cron を使った自動化

systemd の `--watch` オプション によるリアルタイム監視だけでは、高負荷時にファイル追加を見逃す可能性があります。Cron で定期的に未処理ファイルをチェックすることで、見逃しを補完できます。

### Cron設定手順

1. **処理スクリプトの作成**（例：`process_directories.sh`）：

```bash
#!/bin/bash

# メディアディレクトリのベースパス
BASE_DIR="/path/to/base/directory"
# textify スクリプトの配置ディレクトリ
SCRIPT_DIR="/path/to/textify"
# 仮想環境の有効化
source /path/to/textify-env/bin/activate

# メディアを格納する各ディレクトリを処理
ls -dtr ${BASE_DIR}/* | grep -v "log" | while read dir; do
    if [[ -d ${dir} ]]; then
        ${SCRIPT_DIR}/textify.py \
            --input-dir "${dir}" \
            --log-file ${BASE_DIR}/logs/
        sleep 1
    fi
done

# 常にチェックしたい特定ディレクトリも追加
${SCRIPT_DIR}/textify.py \
    --input-dir ${BASE_DIR}/special_media_dir \
    --log-file ${BASE_DIR}/logs/
```

2. **スクリプトに実行権限を付与**：

```bash
chmod +x /path/to/process_directories.sh
```

3. **cron ジョブの追加**（`crontab -e` で編集）：

```bash
# 毎時15分に実行
15 * * * * /usr/bin/flock -n /path/to/lockfile /path/to/process_directories.sh 1> /path/to/logs/cron_textify.log 2> /path/to/logs/cron_textify.err
```

### なぜ systemd と Cron を併用するのか？

* **Systemd + `--watch`**：即時処理が可能だが、高負荷時にイベントを見逃す場合がある
* **Cron**：定期チェックで見逃しを補完
* `flock` により同時実行の競合を防止

これにより、連続稼働が必須の環境でも高い信頼性を実現します。

## 開発ワークフロー

モジュラーアーキテクチャにより、textifyに新機能を簡単に追加できます。機能追加時：

### 新機能の追加

* **音声・動画機能**: `textify/media.py` に追加
* **文書・画像機能**: `textify/documents.py` に追加
* **システム監視**: `textify/system.py` に追加
* **ユーティリティ関数**: `textify/utils.py` に追加
* **CLI オプション**: `textify/cli.py` に追加
* **統合ロジック**: `textify/core.py` に追加

### テスト

モジュラー構造により、個別コンポーネントの分離テストが可能です：

```bash
# 全テストを実行
python -m pytest tests/

# 特定モジュールをテスト
python -m pytest tests/test_media.py
python -m pytest tests/test_documents.py
```

### モジュラー構造の利点

* **単一責任**: 各モジュールは明確で焦点を絞った目的を持ちます
* **保守の容易性**: 特定の機能を素早く特定し変更できます
* **テストの向上**: 個別コンポーネントを分離してテストできます
* **可読性の向上**: 小さく焦点を絞ったファイルは理解しやすくなります
* **結合度の低減**: モジュールは明確に定義されたインターフェースを通じて相互作用します
* **将来の拡張性**: 新しい処理タイプやシステム監視機能を簡単に追加できます

## 作者

[Nobu C. Shirai](https://github.com/nobucshirai)（三重大学）

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) を参照してください。

## 謝辞

* このプロジェクトの基盤となる音声認識システム [Whisper](https://github.com/openai/whisper) を作成しオープンソース化した [OpenAI](https://openai.com) に感謝します。
* このプロジェクトで利用している OCR ライブラリ [EasyOCR](https://github.com/JaidedAI/EasyOCR) を開発し公開している [JaidedAI](https://github.com/JaidedAI) に感謝します。
