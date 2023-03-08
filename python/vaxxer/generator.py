from .features import *


class FeatureGenerator:
    def __init__(
        self,
        seed_fp: str,
        label_fp: str,
        output_dir: str,
        class_label: str = "Multiclass",
        drop_irrelevant: bool = True,
        meta_cols: list = [
            "id_str",
            "usr_id_str",
            "epoch",
            "usr_followers_count",
            "usr_friends_count",
            "usr_favourites_count",
            "usr_statuses_count",
        ],
        train_ratio: float = 0.7,
        verbose: bool = True,
        tweet_filter=None,
    ):
        self.verbose = verbose
        self.seed_fp = seed_fp
        self.label_fp = label_fp
        self.output_dir = output_dir
        self.label_indexer = LabelIndexer()
        if class_label in ["Multiclass"] + self.label_indexer.labels:
            self.class_label = class_label
        else:
            raise ValueError(
                "Choose 'Multiclass' or from %s" % str(self.label_indexer.labels)
            )
        self.drop_irrelevant = drop_irrelevant
        self.meta_cols = meta_cols
        self.train_ratio = train_ratio
        if tweet_filter != None:
            self.tweet_ids = list(pd.read_csv(tweet_filter)["id_str"])
        else:
            self.tweet_ids = None
        self._clear_cache()
        self._load()

    def _clear_cache(self):
        self.tr_text = None
        self.te_text = None
        self.tr_meta = None
        self.te_meta = None
        self.tr_label = None
        self.te_label = None

    def _transform_labels(self, tr_label: list, te_label: list):
        if self.class_label == "Multiclass":
            self.tr_label = tr_label
            self.te_label = te_label
        else:
            label_id = self.label_indexer.label2idx(self.class_label)
            self.tr_label = (np.array(tr_label) == label_id).astype("int")
            if te_label is None:
                self.te_label = None
            else:
                self.te_label = (np.array(te_label) == label_id).astype("int")

    def _load(self):
        start = time.time()
        (
            self.tr_text,
            tr_label,
            self.tr_meta,
            self.te_text,
            te_label,
            self.te_meta,
            unlabeled_text,
            unlabeled_meta,
        ) = get_train_test_data(
            self.seed_fp,
            self.label_fp,
            train_ratio=self.train_ratio,
            meta_cols=self.meta_cols,
            drop_irrelevant=self.drop_irrelevant,
            visualize=False,
            verbose=self.verbose,
            tweet_ids=self.tweet_ids,
        )
        if len(self.te_meta) == 0:
            self.te_text = unlabeled_text
            self.te_meta = unlabeled_meta
            te_label = None
            print("Unlabaled data is set for the test set!")
        self._transform_labels(tr_label, te_label)
        self._export_labels_and_meta()
        print("data loading:", time.time() - start, "seconds\n")

    def __str__(self):
        return "%s_di%s_tr%.2f" % (
            self.class_label,
            self.drop_irrelevant,
            self.train_ratio,
        )

    def _export_labels_and_meta(self):
        label_dir = os.path.join(self.output_dir, str(self), "Labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        np.savetxt(os.path.join(label_dir, "train_label.txt"), self.tr_label)
        print(self.tr_label.shape)
        if not self.te_label is None:
            np.savetxt(os.path.join(label_dir, "test_label.txt"), self.te_label)
            print(self.te_label.shape)
        self.tr_meta.to_csv(os.path.join(label_dir, "train_meta.csv"), index=False)
        self.te_meta.to_csv(os.path.join(label_dir, "test_meta.csv"), index=False)
        print(self.tr_meta.shape)
        print(self.te_meta.shape)

    def _create_folders(self, component: Feature, setting_str: str):
        splitted_component = setting_str.split("_")
        component_type = splitted_component[0]
        if len(splitted_component) > 1:
            component_params = "_".join(splitted_component[1:])
            export_dir = os.path.join(
                self.output_dir, str(self), component_type, component_params
            )
        else:
            export_dir = os.path.join(self.output_dir, str(self), component_type)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        return export_dir

    def _export_numpy_component(self, X_train, X_test, columns, export_dir):
        train_df = pd.DataFrame(X_train, columns=columns)
        train_df.to_csv(os.path.join(export_dir, "train.csv"), index=False)
        test_df = pd.DataFrame(X_test, columns=columns)
        test_df.to_csv(os.path.join(export_dir, "test.csv"), index=False)
        print(X_train.shape)
        print(X_test.shape)
        print(len(columns))

    def _export_torch_component(self, X_train, X_test, export_dir):
        input_ids, masks = X_train
        te_input_ids, te_masks = X_test
        np.savetxt(os.path.join(export_dir, "train_ids.txt"), input_ids.numpy())
        np.savetxt(os.path.join(export_dir, "train_masks.txt"), masks.numpy())
        np.savetxt(os.path.join(export_dir, "test_ids.txt"), te_input_ids.numpy())
        np.savetxt(os.path.join(export_dir, "test_masks.txt"), te_masks.numpy())
        print(input_ids.shape)
        print(masks.shape)
        print(te_input_ids.shape)
        print(te_masks.shape)

    def generate(self, component: Feature, parameters: dict):
        X_train, X_test, columns, setting_str = component.generate(parameters)
        print(setting_str)
        export_dir = self._create_folders(component, setting_str)
        if isinstance(component, BertComponent):
            self._export_torch_component(X_train, X_test, export_dir)
        else:
            self._export_numpy_component(X_train, X_test, columns, export_dir)
