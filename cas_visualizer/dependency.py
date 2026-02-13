from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cassis import Cas
from udapi.core.document import Document
from udapi.core.node import Node
from spacy.displacy import DependencyRenderer

from cas_visualizer._base import Visualizer, VisualizerException

@dataclass
class DependencyFeatureConfig:
    """Configuration for extracting POS and dependency information from CAS annotations.
    
    This class specifies the feature names used to extract linguistic information
    from the CAS type system. Defaults are tailored for DKPro-compatible type systems.
    
    Attributes:
        pos_value: Feature name for part-of-speech tags (e.g., "PosValue" on POS annotations)
        dep_governor: Feature name referencing the governor node in dependency relations
        dep_dependent: Feature name referencing the dependent node in dependency relations
        dep_label: Feature name for the dependency relation label type (e.g., "DependencyType")
    """
    pos_value: str = "PosValue"
    dep_governor: str = "Governor"
    dep_dependent: str = "Dependent"
    dep_label: str = "DependencyType"

@dataclass
class DependencyVisualizerConfig:
    """Configuration for dependency tree visualizers.
    
    Specifies which CAS types and features to use when constructing dependency
    tree visualizations. Supports per-type feature overrides.
    
    Attributes:
        dep_type: Fully qualified CAS type name for dependency relation annotations
        pos_type: Fully qualified CAS type name for POS tags
        span_type: Fully qualified CAS type name for sentence/span boundaries
        feature_config: Feature name mappings (defaults to DKPro conventions)
        feature_map: Optional per-type feature overrides
                    Format: {type_name: {feature_name: override_name}}
    """
    dep_type: str
    pos_type: str
    span_type: str
    feature_config: DependencyFeatureConfig = field(default_factory=DependencyFeatureConfig)  
    feature_map: Optional[Dict[str, Dict[str, str]]] = None

class UdapiDependencyVisualizer(Visualizer):
    def build(self, cas: Cas, *, start: int = 0, end: int = -1) -> list[dict]:
        """
        Build displaCy specs for all sentence spans within [start, end].

        Returns:
        - list of dicts, one per sentence: {'words': [...], 'arcs': [...]}

        Errors:
        - VisualizerException if no sentence spans found or (in strict mode) no tokens/arcs.
        """
        if end == -1:
            end = len(cas.sofa_string)
        if end >= 0 and start > end:
            raise VisualizerException(f"Given span range [start={start}, end={end}] is not valid.")

        parsed: list[dict[str, Any]] = []
        for sent in cas.select(self._span_type):
            if sent.begin >= start and sent.end <= end:
                struct = self._dep_to_dict(cas=cas, covered=sent)
                parsed.append({"words": struct["words"], "arcs": struct["arcs"]})

        if not parsed:
            raise VisualizerException(f"No spans found for type {self._span_type} in range [{start}, {end}].")

        total_words = sum(len(p.get("words", [])) for p in parsed)
        total_arcs = sum(len(p.get("arcs", [])) for p in parsed)
        if self._strict and (total_words == 0 or total_arcs == 0):
            raise VisualizerException(
                f"DependencyVisualizer: nothing to render (words={total_words}, arcs={total_arcs}) "
                f"in range [{start}, {end}]."
            )

        return parsed

    def render(
        self,
        spec: list[dict[str, Any]],
        *,
        output_format: str = "html",
    ) -> str:
        """
        Render the displaCy spec.

        Supported formats:
        - 'html': returns an HTML fragment combining all sentence specs; wrap to full page if page=True.
        - 'svg': returns a single SVG string; exactly one spec required.

        Errors:
        - VisualizerException on unsupported format or if 'svg' is requested for multiple sentences.
        """
        fmt = output_format.lower()
        renderer = DependencyRenderer(options=self._options)

        if fmt == "html":
            frag = renderer.render(spec, page=False, minify=self._minify)
            return self._wrap_html_page(frag, "DependencyVisualizer") if self._page else frag

        if fmt == "svg":
            if len(spec) != 1:
                raise VisualizerException(
                    f"SVG output supports exactly one spec; got {len(spec)}. "
                    f"Render per sentence (build → choose one), or use output_format='html' for multiple sentences."
                )
            p = spec[0]
            return renderer.render_svg("render_id-0", p["words"], p["arcs"])

        raise VisualizerException(f"Output format {output_format} is not supported.")

    def visualize(
        self,
        cas: Cas,
        *,
        start: int = 0,
        end: int = -1,
        output_format: str = "html",
    ) -> str:
        """Convenience: build + render (returns str)."""
        spec = self.build(cas, start=start, end=end)
        return self.render(spec, output_format=output_format)

    def build_tree(self, cas: Cas, sent: FeatureStructure) -> Node:
        udapi_doc = Document()
        udapi_doc.from_conllu_string(self._cas_to_str(cas, sent))
        if len(udapi_doc.bundles) > 1:
            raise ValueError("Multiple bundles per sentence are not supported.")

        return udapi_doc.bundles[0].get_tree()

    def build_trees(self, cas: Cas) -> list[Node]:
        forest = []
        for sent in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
            forest.append(self.build_tree(cas, sent))
        return forest

    def _cas_to_str(self, cas, sentence):
        
        id_list = []
        form_list = []
        lemma_list = []
        udpos_list = []
        pos_list = []
        morph_list = []
        head_list = []
        rel_list = []
        enhanced_deps_list = []
        misc_list = []

        deps_list = cas.select(T_DEP)

        # NB: we need to filter out empty tokens that have no annotations
        unfiltered_token_list = cas.select_covered(T_TOKEN, sentence)
        token_list = [
            x
            for x in unfiltered_token_list
            if not re.match(r"^\s*$", x.get_covered_text())
        ]

        if len(unfiltered_token_list)!=len(token_list):
            print("some tokens are empty!")
        form_list = [x.get_covered_text() for x in token_list]
        print("form_list %s" %(str(form_list)))

        orig_id_list = [x.xmiID for x in token_list]
        id_list = list(range(1, len(token_list) + 1))

        id_map = dict(zip(orig_id_list, id_list))
        print("id_map:\n %s" %(id_map))
        lemma_list = [
            #
            x.value for x in cas.select_covered(T_LEMMA, sentence)
        ]
        pos_list = [x.PosValue for x in cas.select_covered(T_POS, sentence)]
        morph_list = [x.morphTag for x in cas.select_covered(T_MORPH, sentence)]
        if len(morph_list) == 0:
            morph_list = ["_"] * len(token_list)

        # TODO why like this?
        udpos_list = ["FM"] * len(token_list)

        enhanced_deps_list = ["_"] * len(token_list)
        #misc_list = ["_"] * len(token_list)
        print("id_map %s " %(str(id_map)))
        for token in token_list:
            token_id = token.xmiID
            print("processing token_id %s : %s" %(str(token_id), str(token.get_covered_text())))
            misc_list.append("t_start="+str(token.begin)+"|"+"t_end="+str(token.end))
            dep_matches = []
            for dep in deps_list:
                if (
                    dep.Governor.xmiID not in id_map
                    and dep.Dependent.xmiID not in id_map
                ):
                    pass#sys.stderr.write("oops! both goveror and dep not in id map "+str(dep.Governor.xmiID)+"_"+str(dep.Dependent.xmiID)+"\n")
                else:
                    if dep.Dependent.xmiID == token_id:
                        dep_matches.append(dep)
                        print("dependency is %s" %(str(dep)))
                        # root node (in Merlin) has its own id as head!
                        if dep.Governor.xmiID == token_id:
                            head_list.append(0)
                            rel_list.append("root")
                        else:
                            print("dep.Governor.xmiID %s %s" %(str(dep.Governor.xmiID),dep.Governor.get_covered_text()))
                            head_list.append(id_map[dep.Governor.xmiID])
                            rel_list.append(dep.DependencyType)
                    else:
                        pass
            if len(dep_matches) == 0:
                raise RuntimeError(
                    "No dependency matches for token %s !\n" % token.get_covered_text()
                )

        assert len(udpos_list) == len(token_list)
        assert len(head_list) == len(token_list)
        assert len(head_list) == len(rel_list)
        
        colnames = [
            "id",
            "token",
            "lemma",
            "udpos",
            "pos",
            "morph",
            "head",
            "rel",
            "enhanced_deps",
            "misc",
        ]

        lists = [
            id_list,
            form_list,
            lemma_list,
            udpos_list,
            pos_list,
            morph_list,
            head_list,
            rel_list,
            enhanced_deps_list,
            misc_list
        ]
        
        df = pd.DataFrame(lists, colnames).T

        sent_id_line = "# sent_id = 1"
        s_text_line = "# text = " + re.sub("\n", " ", sentence.get_covered_text())
        df_str = df.to_csv(index=False, header=False, sep="\t")
        conllu_string = sent_id_line + "\n" + s_text_line + "\n" + df_str
        conllu_string = re.sub("\n{2,}", "\n", conllu_string).strip()
        print(conllu_string)
        return conllu_string

class SpacyDependencyVisualizer(Visualizer):
    """
    Dependency visualization inside sentence spans using spaCy’s displaCy.

    Contract:
    - build: produce a list of per-sentence specs [{'words': [...], 'arcs': [...]}]
    - render: 'html' → one HTML string (multiple sentences supported),
              'svg'  → one SVG string (requires exactly one sentence/spec)
    - visualize: build + render (returns str)

    Strict mode:
    - If strict=True and there are no sentences or no tokens/arcs, build() raises VisualizerException.
    """
    def __init__(
        self,
        ts: TypeSystem | Path | str,
        config: DependencyVisualizerConfig,
        *,
        minify: bool = False,
        options: Dict[str, Any] | None = None,
        page: bool = False,
        strict: bool = True,
    ):
        super().__init__(ts)
        self._dep_type = config.dep_type
        self._pos_type = config.pos_type
        self._span_type = config.span_type
        self._minify = minify
        self._options = options or {}
        self._page = page
        self._strict = strict

        self._feature_config = config.feature_config
        self._feature_map = config.feature_map or {}

        # Resolve feature names against the TypeSystem for robust access to attributes
        self._pos_value_feature = self._resolve_feature_name(
            self._pos_type,
            self._feature_map.get(self._pos_type, {}).get("value", self._feature_config.pos_value),
            candidates=["PosValue", "pos", "value", "coarseValue", "PosTag", "Tag"]
        )
        self._dep_governor_feature = self._resolve_feature_name(
            self._dep_type,
            self._feature_map.get(self._dep_type, {}).get("governor", self._feature_config.dep_governor),
            candidates=["Governor", "Head", "gov", "head"]
        )
        self._dep_dependent_feature = self._resolve_feature_name(
            self._dep_type,
            self._feature_map.get(self._dep_type, {}).get("dependent", self._feature_config.dep_dependent),
            candidates=["Dependent", "Child", "dep", "child"]
        )
        self._dep_label_feature = self._resolve_feature_name(
            self._dep_type,
            self._feature_map.get(self._dep_type, {}).get("label", self._feature_config.dep_label),
            candidates=["DependencyType", "Relation", "RelType", "label", "type", "dependency"]
        )

    def _resolve_feature_name(self, type_name: str, preferred: str, candidates: list[str]) -> str:
        # Validate feature existence on the given type; pick the first available name
        t = self.ts.get_type(type_name)
        fnames = {f.name for f in t.features}
        if preferred in fnames:
            return preferred
        for cand in candidates:
            if cand in fnames:
                return cand
        if self._strict:
            raise VisualizerException(
                f"Feature resolution failed for type '{type_name}'. "
                f"Tried preferred='{preferred}' and candidates={candidates}."
            )
        # Non-strict: return preferred; getattr(...) will then yield None during extraction
        return preferred

    def build(self, cas: Cas, *, start: int = 0, end: int = -1) -> list[dict[str, Any]]:
        """
        Build displaCy specs for all sentence spans within [start, end].

        Returns:
        - list of dicts, one per sentence: {'words': [...], 'arcs': [...]}

        Errors:
        - VisualizerException if no sentence spans found or (in strict mode) no tokens/arcs.
        """
        if end == -1:
            end = len(cas.sofa_string)
        if end >= 0 and start > end:
            raise VisualizerException(f"Given span range [start={start}, end={end}] is not valid.")

        parsed: list[dict[str, Any]] = []
        for sent in cas.select(self._span_type):
            if sent.begin >= start and sent.end <= end:
                struct = self._dep_to_dict(cas=cas, covered=sent)
                parsed.append({"words": struct["words"], "arcs": struct["arcs"]})

        if not parsed:
            raise VisualizerException(f"No spans found for type {self._span_type} in range [{start}, {end}].")

        total_words = sum(len(p.get("words", [])) for p in parsed)
        total_arcs = sum(len(p.get("arcs", [])) for p in parsed)
        if self._strict and (total_words == 0 or total_arcs == 0):
            raise VisualizerException(
                f"DependencyVisualizer: nothing to render (words={total_words}, arcs={total_arcs}) "
                f"in range [{start}, {end}]."
            )

        return parsed

    def render(
        self,
        spec: list[dict[str, Any]],
        *,
        output_format: str = "html",
    ) -> str:
        """
        Render the displaCy spec.

        Supported formats:
        - 'html': returns an HTML fragment combining all sentence specs; wrap to full page if page=True.
        - 'svg': returns a single SVG string; exactly one spec required.

        Errors:
        - VisualizerException on unsupported format or if 'svg' is requested for multiple sentences.
        """
        fmt = output_format.lower()
        renderer = DependencyRenderer(options=self._options)

        if fmt == "html":
            frag = renderer.render(spec, page=False, minify=self._minify)
            return self._wrap_html_page(frag, "DependencyVisualizer") if self._page else frag

        if fmt == "svg":
            if len(spec) != 1:
                raise VisualizerException(
                    f"SVG output supports exactly one spec; got {len(spec)}. "
                    f"Render per sentence (build → choose one), or use output_format='html' for multiple sentences."
                )
            p = spec[0]
            return renderer.render_svg("render_id-0", p["words"], p["arcs"])

        raise VisualizerException(f"Output format {output_format} is not supported.")

    def visualize(
        self,
        cas: Cas,
        *,
        start: int = 0,
        end: int = -1,
        output_format: str = "html",
    ) -> str:
        """Convenience: build + render (returns str)."""
        spec = self.build(cas, start=start, end=end)
        return self.render(spec, output_format=output_format)

    def _dep_to_dict(self, cas: Cas, covered: FeatureStructure) -> dict[str, Any]:
        """
        Build a displaCy-compatible structure for one sentence.

        Returns:
        - dict with 'words' (tokens with POS) and 'arcs' (dependencies with direction/labels).
        """
        covered_pos = list(cas.select_covered(self._pos_type, covering_annotation=covered))
        offset_to_index = {p.begin: i for i, p in enumerate(covered_pos)}

        # Extract token text and POS tag using the resolved feature name
        words = [
            {"text": p.get_covered_text(), "tag": getattr(p, self._pos_value_feature, None)}
            for p in covered_pos
        ]

        cbegin, cend = covered.begin, covered.end
        arcs: list[dict[str, Any]] = []
        for d in cas.select(self._dep_type):
            # Access governor, dependent, and label via resolved feature names
            gov = getattr(d, self._dep_governor_feature, None)
            dep = getattr(d, self._dep_dependent_feature, None)
            label = getattr(d, self._dep_label_feature, None)

            # Skip dependency if essential features are missing
            if gov is None or dep is None or label is None:
                continue

            gb = gov.begin
            db = dep.begin
            if (cbegin <= gb < cend) and (cbegin <= db < cend):
                if gb in offset_to_index and db in offset_to_index:
                    start_idx = offset_to_index[gb]
                    end_idx = offset_to_index[db]
                    dir_ = "right" if gb <= db else "left"
                    # Normalize to start <= end; keep direction in 'dir'
                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx
                    # Avoid self-loops
                    if start_idx != end_idx:
                        arcs.append({
                            "start": start_idx,
                            "end": end_idx,
                            "label": label,
                            "dir": dir_,
                        })

        return {"words": words, "arcs": arcs}
