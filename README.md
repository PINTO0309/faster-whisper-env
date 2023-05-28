# faster-whisper-env
An environment where you can try out faster-whisper immediately.

## 1. Docker build
```
docker build -t faster-whisper-env -f Dockerfile.gpu .
```
## 2. Docker run
```
docker run --rm -it \
--gpus all \
-v `pwd`:/workdir \
--device /dev/snd:/dev/snd \
pinto0309/faster-whisper-env:latest
```
or
```
docker run --rm -it \
--gpus all \
-v `pwd`:/workdir \
--device /dev/snd:/dev/snd \
faster-whisper-env
```

## 3. Test
- Microphone
    ```bash
    python transcribe.py --mode mic --model_size large-v2
    ```
- 28m59s mp4 test, Large-V2 beam_size=5, RTX3070 (RAM:8GB)
    ```bash
    python test.py
    ```
    ```
    Detected language ja with probability 1.00
    [0.00s -> 7.24s] ステレオ震度推定モデルの最適化 としまして 後半のパート2は 実践
    [7.24s -> 11.60s] のデモを交えまして 普段私がどの ようにモデルを最適化して さまざまな
    [11.60s -> 15.04s] フレームワークの環境へデプロイ してるかというのを 実際に操作
    [15.04s -> 18.28s] をこの画面上で見ていただきながら ご理解いただけるように努めたい
    [18.28s -> 22.12s] と思います それでは早速ですが こちらのGitHub
    [22.12s -> 26.32s] のほうに 本日の講演内容について は 全てチュートリアルをまとめて
    [26.32s -> 32.44s] コミットしてあります 2021-02-28 Intel Deep Learning Day HeatNet
    [32.44s -> 36.28s] デモという ちょっと長い名前なんですけ れども 現状はプライベートになって
    [36.28s -> 40.56s] ますが この講演のタイミングで パブリックのほうに変更したい
    [40.56s -> 43.32s] と思っております 基本的には こちらの上から順番
    [43.32s -> 49.28s] にチュートリアルをなぞっていく という形になります まず 本日
    [49.28s -> 53.52s] 対象にするモデルの内容なんですけ れども Googleリサーチが公開している
    [53.52s -> 58.48s] HeatNetというステレオ震度推定 モデルになります ステレオ震度
    [58.48s -> 62.48s] 推定って何ぞやという話なんですけ れども こういう一つのカメラに
    [62.48s -> 69.64s] 二つのRGBのカメラが付いている タイプの撮影機器を使って 左目
    [69.64s -> 73.64s] と右目の両方から画像を同時に 取得して記録していくと そういう
    [73.64s -> 78.28s] シチュエーションにおいて 2枚の 画像を同時にモデルに入力する
    [78.28s -> 84.16s] と このようにきれいな震度推定 結果が取得できると そういうモデル
    [84.16s -> 89.56s] になります 次に環境ですが 私の 普段で使っているメインの端末
    [89.56s -> 99.52s] がUbuntuの20104で8664という環境です ただ Windows上でも一緒にどっか
    [99.52s -> 104.92s] コンテナ化して作業を進めていきます ので おそらくWSL2などを使用すれば
    [104.92s -> 109.40s] 問題なく動くかと思います あと OpenVINOなんですが インストーラー
    [109.40s -> 115.96s] としてシェアされているものではなくて 今回はちょっと特殊なOpenVINO
    [115.96s -> 120.32s] を直接ビルドして動かしてみる ということまでやってしまいます
    [120.32s -> 125.76s] あとはバックエンドで少し変換 過程で使うOnyxですね 他にもいろいろ
    [125.76s -> 129.16s] もろもろとその環境の中に入って いるんですが 基本的にはコンテナ
    [129.16s -> 133.64s] の中に全ての必要なものが ほとんどのものが導入された状態
    [133.64s -> 137.60s] になってますので あまり皆さんは 気にする必要はなく 作業上から
    [137.60s -> 145.00s] 順番に辿ることができるかと思います 全体の流れですけれども まずHIT
    [145.00s -> 149.16s] NETのモデルがGoogleリサーチから 提供されているんですが TensorFlow
    [149.16s -> 152.88s] のプロトコルバッファーという 一世代前といっても数年前ですね
    [152.88s -> 156.24s] で もうほぼこれディスコになって しまっている形式なんですけれども
    [156.24s -> 159.64s] そのプロトコルバッファーの形式 提供されていますので それを一度
    [159.64s -> 163.28s] TensorFlowのバージョン2の形式である セーブドモデル 推奨されている
    [163.28s -> 167.88s] 形式のほうに変更をかけまして その後 ここがちょっとトリッキー
    [167.88s -> 172.76s] なんですが TensorFlow Liteの形式に 一度最適化のために変換をします
    [172.76s -> 179.28s] その後 Onyxに変換をかけて さらに OpenVINO IR こういった流れで変換
    [179.28s -> 183.84s] をかけてきます 当然 Onyxに途中で 変換したりしてますので その後
    [183.84s -> 189.92s] TensorRTの変換へも簡単にできます あと 私が提供している他のツール
    [189.92s -> 194.68s] をうまく活用すると TensorFlow Lite から 例えば Onyx TensorFlow Liteから
    [194.68s -> 197.76s] OpenVINO それ以外のフォーマット にも もろもろにも変換できます
    [197.76s -> 205.48s] し Onyxから逆向きの変換もできる と そういう手段もご用意しております
    [205.48s -> 211.64s] 大きく1番から9番までのこちら の手順に基づいて実施していきます
    [211.64s -> 214.56s] まずはヒットネットのモデルを ダウンロードしてきて セーブド
    [214.56s -> 220.48s] モデルに変換して Onyxに変換して OpenVINOそのものをビルドします
    [220.48s -> 226.28s] その後で自前でビルドしたOpenVINO を使用して OpenVINO IRというOpenVINO
    [226.28s -> 232.40s] の独自の形式に変換をかけまして 今回はヒットネットのステレオ
    [232.40s -> 236.56s] 振動推定用の テスト用のデータセット ですね MITライセンスで公開されている
    [236.56s -> 240.84s] いいものがありましたので そちら をダウンロードしてきて それを使った
    [240.84s -> 245.36s] デモを3種類 OnyxのデモとOpenVINO のデモとTensorRTのデモと この3
    [245.36s -> 249.40s] パターンをご紹介したいと思います それでは早速ですが手順のほう
    [249.40s -> 252.14s] に入っていきたいと思います まずはGoogleリサーチが公開して
    [252.14s -> 256.32s] くれているヒットネットのモデル の本体をダウンロードしていきます
    [256.32s -> 259.44s] このチュートリアル全体を通して こちらに表示されているコマンド
    [259.44s -> 266.28s] 1行1行コンソールに入力していく と 正常に実行できるはずにして
    [266.28s -> 270.64s] あります モデルは3種類ありまして 一応 どれを選んでもデモまで
    [270.64s -> 274.64s] 到達できるにはしてあるんですが 今回は一番下のRGBハードは2枚
    [274.64s -> 279.64s] を渡すモデルを選択してみたい と思います これ 普段よくある
    [279.64s -> 305.64s] 3チャンネルRGBではなくてRGB RGB の6チャンネルというモデルになります
    [305.64s -> 309.24s] 正常にダウンロードできました では 一度ダウンロードしたモデル
    [309.24s -> 312.92s] をNetronというモデルの構造を可視化 するために便利なツールがあります
    [312.92s -> 318.16s] ので そちらを使用して一度見て みたいと思います こちらはリンク
    [318.16s -> 322.12s] をクリックすると このように簡素 な画面が披露してくるんですけ
    [322.12s -> 325.04s] れども オープンモデルというボタン を押しますと ファイルを選んで
    [325.04s -> 329.04s] ねと聞いてきますので 先ほどダウンロード したプロトコルバッファーを指定します
    [329.04s -> 333.44s] 表示に時間かかりますので もう すでに表示済みのものがこちら
    [333.44s -> 343.12s] になります 入り口がここですね インプットという名前でNHWのところ
    [343.12s -> 347.76s] が全てアンノウンになってますね 全体として かなり大きなモデル
    [347.76s -> 351.64s] で複雑なモデルになってます これ 序盤しか見えてないんですが これを
    [351.64s -> 358.80s] 実際に引きで見ると こんな感じ で ずいっと かなりの数のオペレーション
    [358.80s -> 364.76s] が積み上がっています ようやく ここが出口ですね アウトプット
    [364.76s -> 371.04s] の名前はReference Output Disparity ということみたいです このプロトコル
    [371.04s -> 375.84s] バッファーのファイルは最初しか 使えません ここまで冗長な形式
    [375.84s -> 382.32s] になってますので これを最適化 していきたいと思います
    [382.32s -> 386.28s] では 次にプロトコルバッファー をTensorFlow V2の形式であるセーブド
    [386.28s -> 390.72s] モデルのほうに変換していきたい と思います 普段 皆さん かなり
    [390.72s -> 394.08s] 手こずられるポイントなのかもし れませんけども まず こういうモデル
    [394.08s -> 399.44s] を例えば加工したりだとか 何かの フレームワークに適合させて アプリケーション
    [399.44s -> 402.56s] に組み込んで実行するために まずは 地味に動かしてみたいとか そういう
    [402.56s -> 405.68s] シチュエーションのときに環境 を作ることにかなり苦戦されるん
    [405.68s -> 408.76s] じゃないかなと いろいろと普段 から何かやられてるんじゃない
    [408.76s -> 411.72s] かなと思うんですけれど 事前に 導入されているツールとバージョン
    [411.72s -> 415.20s] が競合してうまく入らない 動かない みたいなことが多々あると思います
    [415.20s -> 419.92s] 私も多分に漏れずそういう状況 によく遭遇しますので 私の独自
    [419.92s -> 422.60s] のDockerコンテナというのを用意 しております 今日は全てその
    [422.60s -> 427.00s] Dockerコンテナ上で作業を完結させる 想定です 一応 そのコンテナという
    [427.00s -> 431.20s] のが どのようなものが導入されている かというのは こちらに記載されて
    [431.20s -> 435.20s] まして 主要なフレームワーク TensorFlowだとかPyTorchだとか あとTensorRT
    [435.20s -> 438.76s] だとかOpenVINOだとか そういった ものは全て導入済みのかなり
    [438.76s -> 444.24s] 大きなコンテナになります 裏を 返しますと ほぼ全てのフレームワーク
    [444.24s -> 449.56s] が入っていますので 何かしら モデルを加工したいみたいな話
    [449.56s -> 451.72s] がありましたら このコンテナ とりあえず起動しておけば何とかな
    [451.72s -> 462.28s] ってしまうと そういう次元のもの ですね では コンテナを起動します
    [462.28s -> 466.76s] はい 起動しました 一応GPUが使える 状態 あとGUIが使える状態のオプション
    [466.76s -> 473.80s] 付きで起動しております こちらで ミドルバーリンのモデルの名前
    [473.80s -> 480.56s] を一旦 環境変数のほうに設定します 続いて PBtoSavedModelという 私 独自
    [480.56s -> 484.76s] のツールなんですが 手軽にプロトコル バッファーをSavedModelの形式で
    [484.76s -> 487.84s] 変換できるためのツールになります プロトコルバッファーのファイル名
    [487.84s -> 491.88s] と あと インプットの言い口の オペレーションの名前ですね あと
    [491.88s -> 494.52s] アウトプットのリファレンスアウトプット ディスパーティーというアウトプット
    [494.52s -> 497.96s] の名前を指定して あとは これオプション なんですけど 出力先のパスを指定
    [497.96s -> 505.00s] してあります これを実行すると 一瞬でSavedModelが生成されます
    [505.00s -> 510.32s] 裏でテストフローが動きまして Optimized Graph Converted SavedModelということで
    [510.32s -> 518.28s] 実際にファイルができ上がっている のを確認できると思います
    [518.28s -> 528.60s] ミドル代わり SavedModel できてますね 一応 GPUを積んでない環境ですとか
    [528.60s -> 532.76s] GUIでわざわざ表示を試す必要もない よという場合は もう少しオプション
    [532.76s -> 536.52s] を減らした状態で 最小環境で起動 するためのコマンドの例もこちら
    [536.52s -> 541.00s] に記載しております 基本的には ほとんど同じです では 生成された
    [541.00s -> 543.76s] SavedModelがどういう形式になっている かというのを テンサフローの標準
    [543.76s -> 547.32s] で付属されているSavedModel CLIという コマンドを使いまして 一度確認
    [547.32s -> 557.00s] してみます 先ほど見ていただいた SavedModel形式のものが 確かにテンサフロー
    [557.00s -> 561.04s] で読み込めましたと 正常に読み 込めた上で 入力と出力がどのような
    [561.04s -> 564.32s] 形式になっているかというのが コンソール上に表示されました
    [564.32s -> 567.24s] 最初にご確認いただいたとおり プロトコル パフォーマーと同じ形式で
    [567.24s -> 572.36s] NHWの部分がアンノウンになっています こちらがアンノウンになっている
    [572.36s -> 577.00s] ので 出力部分はほぼ全てアンノウン になっています こちらをサイズ
    [577.00s -> 580.08s] を固定化してあげることで 内部 の構造はかなり最適化することが
    [580.08s -> 588.32s] できます では 続いて SavedModelから 今度は
    [588.32s -> 593.64s] Onyxに変換します Onyxに変換する 意味としては OpenVinoに変換する
    [593.64s -> 599.60s] ときに Onyx自体はNCHW形式で OpenVino もNCHW形式を基本としております
    [599.60s -> 604.28s] ので 一度 Onyxに変換するという 点が一つ もう一つが Onyxに変換
    [604.28s -> 608.84s] しておくと 他のフレームワーク に対してかなり流動的に変換が
    [608.84s -> 613.56s] 可能です なので ストレートにOpenVino に変換することも可能なんですけ
    [613.56s -> 618.72s] れども 一度 SavedModelからOnyxに変換 しておいて 必要であれば 他のフレームワーク
    [618.72s -> 624.08s] へも横展開するということをします では Dockerコンテナに導入されている
    [624.08s -> 629.00s] SavedModelToTFlightというツールを 使用しまして 名前にちょっとそご
    [629.00s -> 631.84s] わないんですけども このツール 自体が内部でいろんなフォーマット
    [631.84s -> 635.00s] に変換かけられるようになって まして TFlight テンサフローライト
    [635.00s -> 639.84s] にも変換できますし Onyxにも変換 できると そういう代物になります
    [639.84s -> 643.52s] まずは Onyxをストレートに変換する 前に こちらのコマンドで テンサ
    [643.52s -> 648.72s] フローライトのFloat32のモデルを 生成します なぜFloat32を生成する
    [648.72s -> 653.84s] かというと テンサフローライト のオプティマイダーの最適化の動き
    [653.84s -> 658.04s] はとても効率が良くて ストレート にOnyxに変換するよりも 一度テンサ
    [658.04s -> 661.60s] フローライトの形式に変換して あげるほうが 最終的なフォーマット
    [661.60s -> 667.48s] の最適化具合はとても上がります ここから私が作ったツールの動作
    [667.48s -> 678.88s] になります 少し待ちがありますが もともとこのモデルが中がかなり
    [678.88s -> 683.04s] 複雑なので 少し待たされました けれども もう少し軽量なモデル
    [683.04s -> 687.48s] ですと 一瞬で処理が終わります 一応 ログとしてUnknownだったところ
    [687.48s -> 692.80s] に指定した縦横の幅が自動的に 設定されて モデルが生成されました
    [692.80s -> 696.76s] よというログが出ましたね 実際に 生成されたかどうかを確認して
    [696.76s -> 701.04s] みます こちらの別のフォルダー に テンサフローライトが生成されて
    [701.04s -> 709.60s] います 表示を確認してみます 確かにテンサフローライトの形式
    [709.60s -> 714.40s] で読み込みができましたね アウトプット のところが全てUnknownになっていた
    [714.40s -> 718.24s] ところが指定した縦横の幅に合わせて 最適化の上 ちゃんと次元が定義
    [718.24s -> 721.80s] されているという状態を確認できます モデル全体の構造を見ていただいて
    [721.80s -> 725.36s] も分かるとおり 先ほどよりも若干 最適化が進んでますね オペレーション
    [725.36s -> 733.24s] の数はかなり半分ぐらいまで減 ってるんじゃないかなと思います
    [733.24s -> 736.56s] 前半のときにご説明しましたとおり テンサフローライトはかなり
    [736.56s -> 741.40s] オペレーションの融合がかなり得意 なので 数的には多分半分ぐらい
    [741.40s -> 744.44s] 先ほど申し上げましたとおり 半分ぐらいの減っていて 処理自体
    [744.44s -> 748.24s] もかなり効率的になっているん じゃないかなと思います 一度テンサ
    [748.24s -> 751.00s] フローライトは中継地点として ファイルを生成するだけにとどめ
    [751.00s -> 754.32s] まして そのまま今度はテンサフロー ライトのファイルを使用してオニキス
    [754.32s -> 758.68s] を生成します 今度は別のツール を使います テンサフローオニキス
    [758.68s -> 763.12s] というMicrosoftさんが公式化提供 していただいているツールになります
    [763.12s -> 766.96s] こちらにテンサフローライトの ファイルを入力として与えて オニキス
    [766.96s -> 771.60s] のファイルを生成します テンサ フローライトはNHWC形式なんですが
    [771.60s -> 778.16s] オニキスはNCHW形式は基本形式 になりますので Input as NCHWという
    [778.16s -> 784.96s] パラメータを指定してあげて NHWCからNCHW形式へコンバートします
    [784.96s -> 794.20s] モデルのサイズが少し大きいので 若干待たされますが そこまで大きな
    [794.20s -> 801.72s] 待ち時間ではないです 変換が 正常に終了したようです 確かに
    [801.72s -> 804.80s] こちらにテンサフローライトから オニキス形式にコンバートかかった
    [804.80s -> 811.84s] 状態のものは存在しません 単純にテンサフローライトから
    [811.84s -> 815.92s] オニキスへ変換した状態ですと 確かにNCHWの形式変わってはいる
    [815.92s -> 820.12s] ものの こちらを見ていただくと分かるとおり 公式のツールが2枚
    [820.12s -> 826.60s] 1枚1でして 各オペレーションの 入出力の情報が欠けていたり 全体
    [826.60s -> 831.24s] を一つ一つ見ていくと分かるんですけど こういった部分ですね 割と冗長
    [831.24s -> 836.28s] な部分がまだまだ残っております 全体の構造的にはあまり美しくない
    [836.28s -> 843.96s] 状態ですので これをもう一段 最適化しにいきます
    [843.96s -> 846.96s] オニキスシンプリファーという これはサードパーティー製のUCが
    [846.96s -> 850.20s] 作られてるツールなんですけど こちらにオニキスのファイルを
    [850.20s -> 859.12s] 与えてあげて もう一度同じオニキス にかぶせてあげます モデルの構造
    [859.12s -> 873.60s] が少し複雑ですので 多少待ち時間 がかかります
    [873.60s -> 879.92s] ファイルの名前は同じものに 上かぶせしてますので 同じオニキス
    [879.92s -> 884.56s] ファイルを見てみます 最初に生成 したときは 先ほど多分スルーして
    [884.56s -> 887.64s] しまったんですけど おそらく8メガ 前後だったと思います それが82
    [887.64s -> 890.76s] メガというふうにちょっと大きく なってしまっているんですけど
    [890.76s -> 896.84s] 構造はどうなってるか一度見て みます ファイル大きくなっちゃって
    [896.84s -> 898.88s] パフォーマンス落ちるんじゃない かと懸念されている方もいるか
    [898.88s -> 901.96s] と思うんですが 実際に実行して みると分かるんですけれども パフォーマンス
    [901.96s -> 904.44s] にはほとんど影響がありません ただファイルサイズが大きくなって
    [904.44s -> 908.68s] こういった人間が目で見て分かり やすいような不随の情報が孵化
    [908.68s -> 913.28s] された状態で なおかつモデルの 構造も先ほどの少しだけちょっと
    [913.28s -> 916.48s] 分かりにくいですね 今回のパターン は分かりにくいんですが 最適化
    [916.48s -> 927.04s] がされているという状態になります
    [927.04s -> 933.80s] では OnyxからOpenVINOへ変換をする 前に OpenVINOの現状を最新で公開
    [933.80s -> 938.32s] されているインストーラーは実は 中身に少し問題がありまして そちら
    [938.32s -> 942.60s] の問題を解消するために私が自ら インテルのエンジニアの方にイッシュ
    [942.60s -> 947.52s] を挙げまして 問題があるので修正 お願いしますということで こちら
    [947.52s -> 951.16s] のイッシュのほうに投稿しました それが解消されたということ
    [951.16s -> 955.20s] で半年ぐらいかかったんですけれども そのコミットを利用して 一度
    [955.20s -> 960.08s] OpenVINOを最新の状態でビルドを かけまして 最新のモデルオプティマイザー
    [960.08s -> 963.52s] を使用して最適化していきます こちらのコマンド一つなぎになって
    [963.52s -> 968.96s] おりますので 全てコピーしてOpenVINO をビルドします 少し時間かかります
    [968.96s -> 973.60s] ので こちらは一度動画を省略しまして 生成後の状態からもう一度再開
    [973.60s -> 979.00s] させていただきます ビルドが終わり ました これでOpenVINOが全てビルド
    [979.00s -> 986.24s] された状態になっているはずです 期待値としては ビルドされたOpenVINO
    [986.24s -> 989.52s] がホイールファイルになっていて Pythonから気軽に叩けるような状態
    [989.52s -> 992.76s] になっていると嬉しいですので 確かにホイールファイルが生成
    [992.76s -> 998.04s] されているかどうかを確認します 二種類のホイールファイルが生成
    [998.04s -> 1001.88s] されていますね ちょっとファイル名 が長いものと少し短めのもの 一応
    [1001.88s -> 1004.48s] オプティマイザーなどのデベロッパー ツールが含まれているホイール
    [1004.48s -> 1008.88s] が下のほうで 上のほうはインフェレンス エンジンとかが入っているものですね
    [1008.88s -> 1014.12s] というふうに進み分けがされています では 生成されたOpenVINOをDockerコンテナ
    [1014.12s -> 1020.44s] のほうにインストールしていきます 最初から このコンテナはかなり
    [1020.44s -> 1023.08s] 大きなコンテナというご説明を してまして OpenVINOもインストール
    [1023.08s -> 1026.68s] 済みの状態にはなってはいるんです けれども 新たにビルドし直した
    [1026.68s -> 1031.56s] OpenVINOをこちらのホイールファイル で上書き更新してしまいます こちら
    [1031.56s -> 1036.00s] はバグフィックスのために自分で ビルドしたOpenVINOで上書きアクセス
    [1036.00s -> 1043.24s] をするということですね OpenVINO 自体がかなりバックエンドでたくさんの
    [1043.24s -> 1047.00s] 補助的なツールを使う手前ですね 大量にいろんなツールをインストール
    [1047.00s -> 1054.48s] してくれます インストールが終わりました では 新たにインストールしたOpenVINO
    [1054.48s -> 1061.08s] を使用しまして OnyxからOpenVINO IRファイルを生成します こちらのコマンド
    [1061.08s -> 1067.60s] をひとつなぎになっております ので まとめて実行します 今 最適化
    [1067.60s -> 1076.08s] と変換中です こちらはモデルのサイズが大きい
    [1076.08s -> 1081.32s] ので少し待ち時間がかかりますね もう少し軽量なモデルですと これも
    [1081.32s -> 1090.80s] 一瞬で終わります このオプティマイザー とは別に MyLiad CompilerというOpenCV
    [1090.80s -> 1095.00s] AIキットという よく皆さんご存じ だと思うんですけど 半年前ぐらい
    [1095.00s -> 1098.28s] に発売されたステレオカメラですね あちらに適応させるためのコンパイラー
    [1098.28s -> 1101.68s] ツールが用意されているんですが 実はそちらのほうにもまだ一部
    [1101.68s -> 1105.12s] 問題がありまして 今 私のほうで 意思を挙げて インテリアのエンジニア
    [1105.12s -> 1108.72s] の方に対応いただいている最中です 一応 取り扱わせされているので
    [1108.72s -> 1113.24s] もしかしたら数ヶ月後 長くて半年後 ぐらいには対応されて このステレオ
    [1113.24s -> 1116.52s] 振動付付いてモデルがオーク上 でも実行できるようになるかもし
    [1116.52s -> 1120.44s] れませんね サクセスということで モデルの変換が正常に終了しました
    [1120.44s -> 1125.44s] これでOpenVINOのIRモデルが生成 されているはずです 実際に確認
    [1125.44s -> 1131.16s] いたします OpenVINOというフォルダー を生成するコマンドを打ち込んで
    [1131.16s -> 1136.00s] おりまして その中にFloat32で このようにビンファイルとXMLファイル
    [1136.00s -> 1140.80s] この2枚が重要です マッピングファイル は特に使用いたしません 一応 これも
    [1140.80s -> 1149.20s] ネット論を使用して構造確認する ことができます 見た目はオフィン
    [1149.20s -> 1152.72s] キスのときと大きく変わっており ませんが OpenVINO独自のオペレーション
    [1152.72s -> 1158.88s] に置き換えがかなり大量にされて おります 最適化と言いながらも
    [1158.88s -> 1162.72s] 部分的に最適化されております ので 全体としてはそこまで大きく
    [1162.72s -> 1174.00s] 最適化されてないような状況ですね では 次に ここから先はデモの流れ
    [1174.00s -> 1180.40s] になります テストするためのデータセット して 左目と右目 2枚セットで走行
    [1180.40s -> 1185.84s] 中のドライビングの動画 動画という か静止画のデータセットを公開して
    [1185.84s -> 1189.20s] くださってまして 一応 MITライセンス で公開してくださってます そちら
    [1189.20s -> 1192.84s] の動画をあらかじめ私のリポジトリ のほうにダウンロードしてコミット
    [1192.84s -> 1199.72s] しておりますので それをダウンロード します 一応 左目 右目 震度ということで
    [1199.72s -> 1204.08s] 3種類のデータセットになってます プライベートリポジトリになって
    [1204.08s -> 1207.76s] ますので ちょっとGitHubのほうに プライベートリポジトリからプル
    [1207.76s -> 1211.12s] してくるための認証を今 通して おりませんでしたので 偉いなって
    [1211.12s -> 1214.80s] しまいましたが 事前にダウンロード 済みのデータセットを用意して
    [1214.80s -> 1217.92s] おりますので そちらを使ってご説明 したいと思います ダウンロード
    [1217.92s -> 1221.08s] が成功すると ドライビングステレオ イメージズというフォルダが自動的に
    [1221.08s -> 1229.24s] 作られまして ライト レフト デプス ということで 静止画像がこのように
    [1229.24s -> 1234.40s] 全て展開された状態で落ちてくる ようになってます あと テスト用に
    [1234.40s -> 1239.60s] 最後にデモをするときにMP4の動画 を使いますので ステレオ.mp4という
    [1239.60s -> 1246.96s] テスト用の動画も一緒に落ちてきます では ようやくここでデモを実行
    [1247.00s -> 1250.44s] してみたいと思います コンバート の過程でオニキスを設定しました
    [1250.44s -> 1256.76s] ので そちらのオニキスを使って 岩井五郎郎さんが作ってくださった
    [1256.76s -> 1261.76s] オニキスのデモ こちらのリポジトリ を拝借しまして 実行してみたい
    [1261.76s -> 1267.76s] と思います 一応 岩井さんが作って くださったデモから 空打用に最適化
    [1267.76s -> 1271.76s] するために一部 こちらの文字列 を差し替えるためのSDコマンド
    [1271.76s -> 1274.68s] フォークすればいいじゃんという 話ありますけれども 手軽にコピー
    [1274.68s -> 1277.92s] してペースとしてすぐ終わるという 状況ですね 皆さんに実施していただく
    [1277.92s -> 1284.12s] ために 全てSDコマンドで置き換 えております では デモ用のリポジトリ
    [1284.12s -> 1292.12s] をクローンするところと ロジック の書き換えのところを実行しています
    [1292.12s -> 1295.56s] オニキスのランタイムは2種類ありまして オニキスランタイムとオニキス
    [1295.56s -> 1299.36s] ランタイムGPUと2種類あります 今回のモデルは少し重たいモデル
    [1299.36s -> 1304.40s] ですので オニキスランタイムGPU というものにインストールし直して
    [1304.40s -> 1310.96s] おります Gitクローンとランタイムの差し替え
    [1310.96s -> 1314.60s] とプログラムの修正 すべて今 終わりましたので 早速デモを実行
    [1314.60s -> 1324.88s] してみたいと思います 裏では空打が動いてまして 若干
    [1324.88s -> 1331.32s] 遅いかなという感覚は受けます が 20FPS前後出ているんじゃないかな
    [1331.32s -> 1336.84s] と思います 動画が2 3分あります ので 一旦このデモはここで止めます
    [1336.84s -> 1341.32s] ね ただ かなりヒットネット性能 が高いモデルですので 体感的に
    [1341.32s -> 1344.28s] 皆さんどう感じられるか分からない ですが かなりきれいに振動推定
    [1344.28s -> 1347.28s] ができてるんじゃないかなと思います ステレオデプスであるがゆえに
    [1347.28s -> 1352.20s] 計算量は多少単画振動推定よりも 重いんですけれども ここまできれい
    [1352.20s -> 1357.00s] に推定することができますよという 事例ですね 今のがオニキスのデモ
    [1357.00s -> 1366.36s] になります 続いて 私が作成した OpenVINOのモデルのデモになります
    [1366.36s -> 1371.56s] これがカスタムビルドしたOpenVINO を裏で動かして実行すると 標準
    [1371.56s -> 1378.68s] のインストーラーでは実行できない デモになります 先ほどよりもかなり
    [1378.68s -> 1385.36s] 遅いですね 一応 私のマシンがCore i9 第10世代 20スレッドなんですけ
    [1385.36s -> 1389.28s] れど そこまでハイパワーなマシン を使っても これぐらい重たいモデル
    [1389.28s -> 1393.92s] です モデルの内部はコンボリューション の3Dっていうのがかなりたくさん
    [1393.92s -> 1397.92s] ありまして 恐らくCPU推論するときに コンボリューション3Dがネック
    [1397.92s -> 1404.24s] になってるんじゃないかなと思います 23FPSですかね ちょっと重いですが
    [1404.24s -> 1414.92s] 精度的には大差がないですね 続いて 精度は落とさずにかなり
    [1414.92s -> 1420.04s] 早いデモということで 多分 今まで OnyxとOpenVINOよりも圧倒的に
    [1420.04s -> 1425.08s] 早いであろう TensorRTのデモも岩竹 さんという方から許可いただいて
    [1425.08s -> 1435.68s] 借用しております そちらのデモ もご覧いただきたいと思います
    [1435.68s -> 1439.64s] では TensorRT用のリポジトリですが 岩竹さんのリポジトリからクローン
    [1439.64s -> 1445.44s] してきたものの環境で 一度 Dockerコンテナを起動します Dockerコンテナ
    [1445.44s -> 1449.28s] を起動する理由は TensorRTが導入 されている環境をすぐ使いたい
    [1449.28s -> 1466.04s] からですね 同じ場所に来ました 既に私が手元でビルド済みのもの
    [1466.04s -> 1471.36s] がメインという本体のバイナリー になります こちらを使いまして
    [1471.36s -> 1480.64s] デモを実行してみたいと思います どうでしょうか さっきよりも全然
    [1480.64s -> 1485.28s] 早くてですね フレームレートが 30FPS超えてますね 1フレームあたり
    [1485.28s -> 1489.28s] の推論 2フレームの同時推論で 20ミリセックというかなり早い
    [1489.28s -> 1493.28s] 推論結果になってます 若干ちょっと 私のMP4の動画の作り方がよくなか
    [1493.28s -> 1497.56s] って ノイズが入っておりますけれども 基本性能自体は同じOnyx使って
    [1497.56s -> 1509.68s] おりますので 変わりはないという 感じです
    [1509.68s -> 1514.04s] では OnyxとOpenVINOとTensorRT この三種類のデモをさせていただき
    [1514.04s -> 1520.56s] ました これで一通りカスタムビルド のOpenVINOを使った最適化という
    [1520.56s -> 1526.92s] 講演のメインの部分のお話は終了 になります 次は裏話というところ
    [1526.92s -> 1530.40s] も聞きたいされてる方がいるかもし れませんので もう少しですね 今回
    [1530.40s -> 1534.48s] はかなり簡単にモデル変換できる パターンのご紹介だったわけですけ
    [1534.48s -> 1537.88s] れども 普段私がチャレンジしている モデル変換というのはもっと難易度
    [1537.88s -> 1543.24s] が高くて 実際 今日やったような モデル変換は大体15分ぐらいで
    [1543.24s -> 1547.24s] やってるんですけれども 難易度 高いモデルは数時間かかっている
    [1547.24s -> 1551.20s] という感じです その状況をどんな 感じなのかというのを 黒話みたい
    [1551.20s -> 1555.40s] になるんですけども お伝えしたい と思います 裏話ということなんですけど
    [1555.40s -> 1558.24s] ここでコマンドを1から叩いている とめちゃくちゃ時間かかって全然
    [1558.24s -> 1561.28s] 時間が収まらなくなってしまいます ので 普段私が作っているツール
    [1561.28s -> 1564.84s] のほうに上がってきているイシュー で簡単にご説明したいと思います
    [1564.84s -> 1568.76s] OpenVINO2 TensorFlowという私の独自ツール なんですけど これはOpenVINOのIRモデル
    [1568.76s -> 1573.32s] からTensorFlowに逆変換すると ちょっと 一風変わったもので 多分私しか
    [1573.32s -> 1577.28s] 作ってないんじゃないかなと思います そこに君のツールを使って変換
    [1577.28s -> 1586.80s] したらエラーが出たよということ です 確かこれが YORO V5 Liteのモデル
    [1586.80s -> 1593.64s] をOpenVINOからTensorFlow Liteに変換 かけたら ツールからこんなエラー
    [1593.64s -> 1601.44s] が出ちゃうんだよねと 確かに出ます と エラーになる理由が チャンネル
    [1601.44s -> 1605.96s] 変換といって YORO V5の場合は結構 特殊な処理があるんですけど 5次元
    [1605.96s -> 1610.64s] に加工した上で チャンネル部分 をスイッチしてっていうのを さん
    [1610.64s -> 1614.84s] ざんパラコーモデルの中で何段 にも分けてやってるんですが 5次元
    [1614.84s -> 1618.30s] のチャンネルシフトっていうのは そもそも5次元っていうのはツール
    [1618.30s -> 1621.40s] 上でどこからどこへ展示すれば いいかっていうのを予測すること
    [1621.40s -> 1625.56s] がかなり難しくってエラーになる ことを前提としてツール論を設計
    [1625.56s -> 1630.92s] しておりますので 実はそのエラー は5次元ではよく発生するので ツール
    [1630.92s -> 1634.48s] の動作を変えるためのJSONファイル が用意してあるから それを食わ
    [1634.48s -> 1638.24s] せればいいんだよということで 私のほうで提示してます このJSON
    [1638.24s -> 1641.40s] ファイルを作るのがかなり大変 で トライハンドエラーでエラー
    [1641.40s -> 1644.64s] になってはJSONにここの部分 このレイヤーの動きを変えろという
    [1644.64s -> 1649.16s] 指示を出し もう一度実行してエラー になった このオペレーションの
    [1649.16s -> 1653.08s] 動作を 展示動作をこのように変え なさいという指示をツールに出し
    [1653.08s -> 1656.88s] もう一回実行したらエラーになった じゃあ次のオペレーションですね
    [1656.88s -> 1660.28s] ここはリシェープかな リシェープ の動作 この形状に変えなさい
    [1660.28s -> 1664.12s] っていうのを指示を出しっていう のを延々とやっています まだ
    [1664.12s -> 1668.08s] このYORO VGOライトはモデル全体 の構造がすごく小さくてシンプル
    [1668.08s -> 1672.36s] ですので ただこれぐらいで終わ ってるんですけど 例えばステレオ
    [1672.36s -> 1674.96s] 芯の推定モデルのヒットネット はすごくシンプルでしたが それ
    [1674.96s -> 1678.20s] 以外の世に出回っているステレオ 芯の推定モデルってものすごく
    [1678.20s -> 1683.40s] モデルの構造がもっと複雑です そういうものが当然変化確率は
    [1683.40s -> 1687.08s] エラーになるんですけど このJSONファイル はこれしきりではなくて これの
    [1687.08s -> 1690.68s] 50倍ぐらいの長さのJSONファイル になってしまいます ツールで
    [1690.68s -> 1695.24s] 自動的に変換できれば嬉しいんですが 今 例えばマイクロソフトさん
    [1695.24s -> 1700.00s] だとかGoogleさんが提供してくれている 公式のツールでは変換ができない
    [1700.00s -> 1704.12s] もの 変換できたとしても 最適化 が不十分なまま変換されるもの
    [1704.12s -> 1707.72s] っていうのが大多数です そこで 私のツールであれば ツールその
    [1707.72s -> 1712.20s] ものの動作をJSONファイルで書き換える ことができますので ほとんどの
    [1712.20s -> 1718.12s] パターン LSTM以外の画像系のモデル であれば ほぼ確実に変換すること
    [1718.12s -> 1724.28s] ができますと 時間をかければという 状況です 裏話は以上です もっと
    [1724.28s -> 1731.20s] お話聞きたい方は 私のOSSあります ので 一周なので ご質問いただく
    [1731.20s -> 1737.76s] か ディスカッションのほうでお 待ちしております 本日の私の講演
    [1737.76s -> 1739.68s] は以上になります ご清聴ありがとうございました

    :-: main --> 137.2445514202118 sec
    ```

## 4. Models
```
tiny.en
tiny
base.en
base
small.en
small
medium.en
medium
large-v1
large-v2
```
