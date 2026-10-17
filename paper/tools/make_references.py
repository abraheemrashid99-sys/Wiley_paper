"""Write references.tex: the verified records (Wiley Unified style, numbered in
order of first citation). Every record was checked against its publisher,
proceedings or arXiv page; see paper/REFERENCES_CHECK.md."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REF = {
"itu2023": r"International Telecommunication Union, Radiocommunication Sector, ``Framework and Overall Objectives of the Future Development of IMT for 2030 and Beyond,'' Recommendation ITU-R M.2160-0 (ITU, 2023).",
"strinati2021": r"E. C. Strinati and S. Barbarossa, ``6G Networks: Beyond Shannon Towards Semantic and Goal-Oriented Communications,'' \textit{Computer Networks} 190 (2021): 107930.",
"letaief2019": r"K. B. Letaief, W. Chen, Y. Shi, J. Zhang, and Y.-J. A. Zhang, ``The Roadmap to 6G: AI Empowered Wireless Networks,'' \textit{IEEE Communications Magazine} 57, no. 8 (2019): 84--90.",
"saad2020": r"W. Saad, M. Bennis, and M. Chen, ``A Vision of 6G Wireless Systems: Applications, Trends, Technologies, and Open Research Problems,'' \textit{IEEE Network} 34, no. 3 (2020): 134--142.",
"shannon1948": r"C. E. Shannon, ``A Mathematical Theory of Communication,'' \textit{Bell System Technical Journal} 27, no. 3 (1948): 379--423.",
"weaver1949": r"W. Weaver, ``Recent Contributions to the Mathematical Theory of Communication,'' in C. E. Shannon and W. Weaver, \textit{The Mathematical Theory of Communication} (University of Illinois Press, 1949).",
"gunduz2023": r"D. G\"und\"uz, Z. Qin, I. E. Aguerri, et al., ``Beyond Transmitting Bits: Context, Semantics, and Task-Oriented Communications,'' \textit{IEEE Journal on Selected Areas in Communications} 41, no. 1 (2023): 5--41.",
"luo2022": r"X. Luo, H.-H. Chen, and Q. Guo, ``Semantic Communications: Overview, Open Issues, and Future Research Directions,'' \textit{IEEE Wireless Communications} 29, no. 1 (2022): 210--219.",
"kountouris2021": r"M. Kountouris and N. Pappas, ``Semantics-Empowered Communication for Networked Intelligent Systems,'' \textit{IEEE Communications Magazine} 59, no. 6 (2021): 96--102.",
"farsad2018": r"N. Farsad, M. Rao, and A. Goldsmith, ``Deep Learning for Joint Source-Channel Coding of Text,'' in \textit{2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)} (IEEE, 2018), 2326--2330.",
"xie2021deepsc": r"H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, ``Deep Learning Enabled Semantic Communication Systems,'' \textit{IEEE Transactions on Signal Processing} 69 (2021): 2663--2675.",
"bourtsoulatze2019": r"E. Bourtsoulatze, D. Burth Kurka, and D. G\"und\"uz, ``Deep Joint Source-Channel Coding for Wireless Image Transmission,'' \textit{IEEE Transactions on Cognitive Communications and Networking} 5, no. 3 (2019): 567--579.",
"kurka2021": r"D. Burth Kurka and D. G\"und\"uz, ``Bandwidth-Agile Image Transmission With Deep Joint Source-Channel Coding,'' \textit{IEEE Transactions on Wireless Communications} 20, no. 12 (2021): 8081--8095.",
"wang2024llmsc": r"Z. Wang, L. Zou, S. Wei, et al., ``Large-Language-Model-Enabled Text Semantic Communication Systems,'' \textit{Applied Sciences} 15, no. 13 (2025): 7227.",
"qiao2025token": r"L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, ``Token Communications: A Large Model-Driven Framework for Cross-Modal Context-Aware Semantic Communications,'' \textit{IEEE Wireless Communications} 32, no. 5 (2025): 80--88.",
"guo2023importance": r"S. Guo, Y. Wang, S. Li, and N. Saeed, ``Semantic Importance-Aware Communications Using Pre-Trained Language Models,'' \textit{IEEE Communications Letters} 27, no. 9 (2023): 2328--2332.",
"jiang2024largeai": r"F. Jiang, L. Dong, Y. Peng, et al., ``Large AI Model Empowered Multimodal Semantic Communications,'' \textit{IEEE Communications Magazine} 63, no. 1 (2025): 76--82.",
"xie2025largemodel": r"H. Xie, Z. Qin, X. Tao, and Z. Han, ``Toward Intelligent Communications: Large Model Empowered Semantic Communications,'' \textit{IEEE Communications Magazine} 63, no. 1 (2025): 69--75.",
"bariah2024": r"L. Bariah, Q. Zhao, H. Zou, Y. Tian, F. Bader, and M. Debbah, ``Large Generative AI Models for Telecom: The Next Big Thing?,'' \textit{IEEE Communications Magazine} 62, no. 11 (2024): 84--90.",
"chen2025llmsc": r"M. Chen, Z. Sun, X. He, L. Wang, and A. Al-Dulaimi, ``LLM-Based Semantic Communication: The Way From Task-Originated to General,'' \textit{IEEE Wireless Communications Letters} 14, no. 10 (2025): 3029--3033.",
"ni2025multitask": r"W. Ni, Z. Qin, Y. Feng, et al., ``Multi-Task Semantic Communications via Large Models,'' \textit{IEEE Communications Standards Magazine} 9, no. 4 (2025): 16--23.",
"zhang2023dynamic": r"H. Zhang, S. Shao, M. Tao, X. Bi, and K. B. Letaief, ``Deep Learning-Enabled Semantic Communication Systems With Task-Unaware Transmitter and Dynamic Data,'' \textit{IEEE Journal on Selected Areas in Communications} 41, no. 1 (2023): 170--185.",
"zhang2026kbident": r"Y. Zhang, J. Huang, J. Wang, J. Feng, and Z. Fei, ``Knowledge Base Identification and Adaptation for End-to-End Semantic Communication,'' \textit{IEEE Internet of Things Journal} 13, no. 18 (2026): 42698--42712.",
"zhang2026kbupdate": r"J. Zhang, S. Li, Y. Sun, et al., ``Cooperative Semantic Knowledge Base Update for Semantic Communication Networks,'' \textit{IEEE Transactions on Communications} 74 (2026): 1391--1405.",
"caire2001": r"G. Caire and D. Tuninetti, ``The Throughput of Hybrid-ARQ Protocols for the Gaussian Collision Channel,'' \textit{IEEE Transactions on Information Theory} 47, no. 5 (2001): 1971--1988.",
"jiang2022harq": r"P. Jiang, C.-K. Wen, S. Jin, and G. Y. Li, ``Deep Source-Channel Coding for Sentence Semantic Transmission With HARQ,'' \textit{IEEE Transactions on Communications} 70, no. 8 (2022): 5225--5240.",
"radford2019": r"A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, ``Language Models Are Unsupervised Multitask Learners,'' OpenAI Technical Report (2019).",
"hu2022lora": r"E. J. Hu, Y. Shen, P. Wallis, et al., ``LoRA: Low-Rank Adaptation of Large Language Models,'' in \textit{International Conference on Learning Representations (ICLR)} (2022).",
"behrouz2025": r"A. Behrouz, M. Razaviyayn, P. Zhong, and V. Mirrokni, ``Nested Learning: The Illusion of Deep Learning Architectures,'' in \textit{Advances in Neural Information Processing Systems 38 (NeurIPS 2025)} (2025), 52459--52493.",
"vaswani2017": r"A. Vaswani, N. Shazeer, N. Parmar, et al., ``Attention Is All You Need,'' in \textit{Advances in Neural Information Processing Systems 30} (2017), 5998--6008.",
"koehn2005": r"P. Koehn, ``Europarl: A Parallel Corpus for Statistical Machine Translation,'' in \textit{Proceedings of Machine Translation Summit X: Papers} (2005), 79--86.",
"equitz1991": r"W. H. R. Equitz and T. M. Cover, ``Successive Refinement of Information,'' \textit{IEEE Transactions on Information Theory} 37, no. 2 (1991): 269--275.",
"kusupati2022": r"A. Kusupati, G. Bhatt, A. Rege, et al., ``Matryoshka Representation Learning,'' in \textit{Advances in Neural Information Processing Systems 35} (2022), 30233--30249.",
"kirkpatrick2017": r"J. Kirkpatrick, R. Pascanu, N. Rabinowitz, et al., ``Overcoming Catastrophic Forgetting in Neural Networks,'' \textit{Proceedings of the National Academy of Sciences of the United States of America} 114, no. 13 (2017): 3521--3526.",
"alistarh2017": r"D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic, ``QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding,'' in \textit{Advances in Neural Information Processing Systems 30} (2017), 1709--1720.",
"mcmahan2017": r"H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Ag\"uera y Arcas, ``Communication-Efficient Learning of Deep Networks From Decentralized Data,'' in \textit{Proceedings of the 20th International Conference on Artificial Intelligence and Statistics}, PMLR 54 (2017), 1273--1282.",
"papineni2002": r"K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, ``BLEU: A Method for Automatic Evaluation of Machine Translation,'' in \textit{Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics} (2002), 311--318.",
"lafferty2001": r"J. D. Lafferty, A. McCallum, and F. C. N. Pereira, ``Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data,'' in \textit{Proceedings of the 18th International Conference on Machine Learning} (2001), 282--289.",
"zhang2015agnews": r"X. Zhang, J. Zhao, and Y. LeCun, ``Character-Level Convolutional Networks for Text Classification,'' in \textit{Advances in Neural Information Processing Systems 28} (2015), 649--657.",
"maatouk2023teleqna": r"A. Maatouk, F. Ayed, N. Piovesan, A. De Domenico, M. Debbah, and Z.-Q. Luo, ``TeleQnA: A Benchmark Dataset to Assess Large Language Models Telecommunications Knowledge,'' \textit{IEEE Network} 40, no. 2 (2026): 253--260.",
"misra2022": r"R. Misra, ``News Category Dataset,'' arXiv:2209.11429 (2022).",
"efron1993": r"B. Efron and R. J. Tibshirani, \textit{An Introduction to the Bootstrap} (Chapman \& Hall, 1993).",
"holm1979": r"S. Holm, ``A Simple Sequentially Rejective Multiple Test Procedure,'' \textit{Scandinavian Journal of Statistics} 6, no. 2 (1979): 65--70.",
"kunsch1989": r"H. R. K\"unsch, ``The Jackknife and the Bootstrap for General Stationary Observations,'' \textit{Annals of Statistics} 17, no. 3 (1989): 1217--1241.",
"schuirmann1987": r"D. J. Schuirmann, ``A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability,'' \textit{Journal of Pharmacokinetics and Biopharmaceutics} 15, no. 6 (1987): 657--680.",
"polyanskiy2010": r"Y. Polyanskiy, H. V. Poor, and S. Verd\'u, ``Channel Coding Rate in the Finite Blocklength Regime,'' \textit{IEEE Transactions on Information Theory} 56, no. 5 (2010): 2307--2359.",
}

files = ["frontmatter.tex", "sec1_introduction.tex", "sec2_related.tex", "sec3_system.tex", "sec4_method.tex", "sec5_setup.tex", "sec6_results.tex", "sec7_conclusion.tex", "endmatter.tex"]
order = []
for name in files:
    for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", (HERE / name).read_text(encoding="utf-8")):
        for key in group.split(","):
            key = key.strip()
            if key not in order:
                order.append(key)
missing = [k for k in order if k not in REF]
unused = [k for k in REF if k not in order]
if missing:
    raise SystemExit(f"cited but not in the verified list: {missing}")
lines = ["% Generated by tools/make_references.py from verified records (Wiley Unified style).", "\\begin{thebibliography}{99}"]
lines += [f"\\bibitem{{{k}}} {REF[k]}" for k in order]
lines.append("\\end{thebibliography}")
(HERE / "references.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(order)} references written; unused verified records: {unused}")
