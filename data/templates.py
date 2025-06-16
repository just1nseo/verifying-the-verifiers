DOCUMENT_PLACEHOLDER = "[DOCUMENT_PLACEHOLDER]"
STATEMENT_PLACEHOLDER = "[STATEMENT_PLACEHOLDER]"

FEW_SHOT_TEMPLATE = f"""\
Determine whether the provided statement is attributable to the document.
1. If any information in the statement is missing in the document, you should say "[Not Attributable]".
2. If any information in the statement is contradictory to the document, you should say "[Contradictory]".
3. If all information in the statement is supported by the document, you should say "[Attributable]".
4. Before showing your answer, think step-by-step and show your specific reasoning. When you extract information from the document, you should use [Extraction] to indicate that. When you infer information using the document, you should use [Inference] to indicate that. When you conclude your reasoning, you should use [Conclusion] to indicate that.
5. You must use the information only in the document to make your decision.

Examples:

### Example 1

Document:
-- that unless there is a -- the Court is -- because the circumstances change, that we are living in an era now where piracy is a significant problem, there's question of administrative ease, of administering a system where copyrights may be different for one set of authors, or different for another set of authors, there's changes that are taking place internationally, so that what we're saying is that not only could this Court conceive of reasons why Congress thought it was accomplishing the objectives of this clause, but that there are numerous objectives that are entirely legitimate in --

Statement:
The discussion is not related to pirates

Answer:
[Extraction] According to the document, piracy is mentioned as a significant problem.
[Extraction] However, the document suggests that "piracy" is likely referring to copyright infringement, rather than the traditional sense of pirates.
[Conclusion] So, the statement is [Attributable].

### Example 2

Document:
Paul Brittain (nephew)
Bill Quinn (father-in-law)

George Robert Newhart (September 5, 1929 – July 18, 2024) was an American comedian and actor. Newhart was known for his deadpan and stammering delivery style. Beginning his career as a stand-up comedian, he transitioned his career to acting in television. He received numerous accolades, including three Grammy Awards, an Emmy Award, and a Golden Globe Award.

Statement:
George Newhart initially began his career in the film industry performing stand-up comedy.

Answer:
[Extraction] According to the document, George Robert Newhart was known for his deadpan and stammering delivery style.
[Extraction] Beginning his career as a stand-up comedian, he transitioned his career to acting in television.
[Inference] This implies that he initially began his career performing stand-up comedy.
[Extraction] However, the document does not state that he initially began his career in the film industry.
[Conclusion] So, the statement is [Not Attributable].

### Example 3

Document:
Scarlet Days is a 1919 American silent western film produced and directed by D. W. Griffith and released through Paramount/Artcraft Pictures, Artcraft being an affiliate of Paramount. Richard Barthelmess stars in a role for which Griffith had screentested Rudolph Valentino. It is considered by many to be one of Griffith's worst films.

Statement:
Richard Barthelmess did not work for Artcraft Pictures

Answer:
[Extraction] According to the document, Scarlet Days is a 1919 American silent western film released through Paramount/Artcraft Pictures.
[Extraction] Richard Barthelmess stars in Scarlet Days.
[Conclusion] So, the statement is [Contradictory].

### Example 4

Document:
The Bakun Hydroelectric Project (BHEP) comprises the construction of a 2,400MW hydroelectric dam, the transmission of its electricity, and the building of related infrastructure including access roads.

Statement:
The Bakun Hydroelectric Project is constructing a 2,400MW hydroelectric dam which will transmit solar power using access roads.

Answer:
[Extraction] According to the document, The Bakun Hydroelectric Project (BHEP) comprises the construction of a 2,400MW hydroelectric dam.
[Extraction] The document also mentions the transmission of its electricity, and the building of related infrastructure including access roads.
[Extraction] However, the document does not state that the project will transmit solar power.
[Inference] The fact that it is a hydroelectric dam implies that it will transmit hydroelectric power, not solar power.
[Conclusion] So, the statement is [Contradictory].

### Example 5

Document:
Murphy is out of the lineup against St. Louis on Tuesday. Murphy will receive a breather following three straight starts, including an 0-for-4 day during the series opener Monday. In his place, Jeff Mathis will catch Zack Greinke and bat eighth in the order.

Statement:
Murphy did not hit a home run on Monday.

Answer:
[Extraction] According to the document, Murphy had an 0-for-4 day during the series opener Monday.
[Inference] This implies that Murphy did not hit a home run on Monday, as he did not get any hits.
[Conclusion] So, the statement is [Attributable].

### Example 6

Document:
Inmate Death<br>An inmate died at Waymart Correctional Facility in Pennsylvania. The inmate slipped and fell and hit his head on the floor. He began bleeding from his head and lay there for ten minutes. When medical arrived, the man was unresponsive. With no oxygen to his brain, the prison did not offer life support.

Statement:
after the fall, the man was unresponsive but had brain activity

Answer:
[Extraction] According to the document, the inmate slipped and fell and hit his head on the floor, began bleeding from his head, and lay there for ten minutes.
[Extraction] When medical arrived, the man was unresponsive.
[Inference] There was no oxygen to his brain, which implies that there was no brain activity.
[Conclusion] So, the statement is [Contradictory].

### Example 7

Document:
She changed her professional name from Lily Allen to Lily Rose Cooper. In August 2013 she changed her professional name back to Allen and tweeted new music would be arriving "soon". Allen initially said that her record label would not allow the release of "Sheezus" as an official single because the song was "not up-tempo enough" and contained the word "period". Allen continued to comment on the matter saying her label wanted to release radio friendly songs such as "Air Balloon", which Allen called "docile pop" and said they were chosen because labels and radio stations "won't play the better stuff". The song contains references to American singer-songwriters Beyoncé, Katy Perry, and Lady Gaga, as well as Barbadian singer Rihanna, and New Zealand singer-songwriter Lorde. Critical reception

Upon release the song was met with polarized reviews from music critics. Jason Lipshutz of Billboard praised the song for being "anti-pop", continuing to call it an "sarcastic pop anthem." Time magazine praised the song and Allen for wanting "her ladies to unite in bringing their A-game." Radhika Sanghani of The Daily Telegraph on the other hand said that " ultimately, Allen’s message itself is confused. It seems like she is trying to join Katy Perry and Queen B in jumping onto the feminist pop movement[...] it’s just a shame that she’s doing it so obviously we can recognise it."

Statement:
"Sheezus" is described as an ironic pop anthem and an anti-pop song.

Answer:
[Extraction] According to the document, Jason Lipshutz of Billboard praised the song for being "anti-pop", continuing to call it an "sarcastic pop anthem."
[Extraction] The document does not state that "Sheezus" is described as an ironic pop anthem.
[Extraction] However, it does state that it is described as a sarcastic pop anthem, which is similar.
[Conclusion] So, the statement is [Attributable].

### Example 8

Document:
US 51 – Wisconsin Rapids, Plover, Waupaca | nan
Wisconsin | Portage | Village of Plover | 147.51 | 237.39 | 153 | CTH-B (Plover Road) – Wisconsin Rapids, Plover, Amherst | nan
Wisconsin | Portage | Village of Plover | 150.53 | 242.25 | 156 | CTH-HH (McDill Avenue) – Whiting, Stevens Point | nan
Wisconsin | Portage | Stevens Point | 152.71 | 245.76 | 158 | US 10 east / WIS 66 west (Main Street) – Stevens Point, Waupaca, Appleton, Marshfield | Southern end of US 10 concurrency; southe
rn end of WI 66 concurrency; signed as exits 158A (east) and 158B (west) northbound
Wisconsin | Portage | Stevens Point | 153.94 | 247.74 | 159 | WIS 66 east (Stanley Street) – Stevens Point, Rosholt | Northern end of WI 66 concurrency
Wisconsin | Portage | Stevens Point | 155.76 | 250.67 | 161 | Bus. US 51 (Division Street) – Stevens Point | nan
Wisconsin | Portage | Hull | 157.63 | 253.68 | 163 | Casimir Road | To CTH-X
Wisconsin | Portage | Hull | 159.75 | 257.09 | 165 | US 10 west – Marshfield | Northern end of US 10 concurrency
Wisconsin | Portage | Hull | 159.75 | 257.09 | 165 | CTH-X | Former diamond interchange; removed for construction of US 10 exit
Wisconsin | Portage | Town of Dewey | 165.39 | 266.17 | 171 | CTH-DB – Knowlton, Lake DuBay | nan
Wisconsin | Marathon | Town of Knowlton | 169.64 | 273.01 | 175 | WIS 34 (Balsam Road) – Knowlton, Wisconsin Rapids | nan
Wisconsin | Marathon | Mosinee | 173.57 | 279.33 | 179 | WIS 153 – Mosinee, Elderon | Central Wisconsin Airport
Wisconsin | Marathon | Kronenwetter | 175.39 | 282.26 | 181 | Maple Ridge Road | nan
Wisconsin | Marathon | Rothschild | 179.52 | 288.91 | 185 | Bus.

Statement:
The route of Interstate 39 passes through Illinois and Wisconsin.

Answer:
[Extraction] According to the document, there is no information about the route of Interstate 39 passing through Illinois.
[Conclusion] So, the statement is [Not Attributable].

### Example 9

Document:
How to deal with annoying teachers<br>Ask them what they are looking for. If your teacher is a hard grader, try to get more details when they give an assignment. Ask if there is anything specific they are looking for, and if there's anything you should avoid.

Statement:
Annoying teachers are the best.

Answer:
[Extraction] According to the document, the document does not state that annoying teachers are the best.
[Conclusion] So, the statement is [Not Attributable].

Now please start the task. You must follow the format strictly. Don't add any additional information.

Document:
{DOCUMENT_PLACEHOLDER}

Statement:
{STATEMENT_PLACEHOLDER}"""

ZERO_SHOT_TEMPLATE = f"""\
Instructions:
1. You have been given a STATEMENT and some DOCUMENT.
2. Determine whether the given STATEMENT is supported by the given DOCUMENT. The STATEMENT does not need to be explicitly supported by the DOCUMENT, but should be strongly implied by the DOCUMENT.
3. Before showing your answer, think step-by-step and show your specific reasoning. As part of your reasoning, summarize the main points of the DOCUMENT.
4. If the STATEMENT is supported by the DOCUMENT, be sure to show the supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either [Attributable] or [Not Attributable], or [Contradictory].
7. Wrap your final answer in square brackets.

DOCUMENT:
{DOCUMENT_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""

CLEARCHECK_COT = f"""\
Determine whether the provided statement is attributable to the document.
1. If any information in the statement is missing in the document, you should say "[Not Attributable]".
2. If any information in the statement is contradictory to the document, you should say "[Contradictory]".
3. If all information in the statement is supported by the document, you should say "[Attributable]".
4. Before showing your answer, think step-by-step and show your specific reasoning. When you extract information from the document, you should use [Extraction] to indicate that. When you infer information using the document, you should use [Inference] to indicate that. When you conclude your reasoning, you should use [Conclusion] to indicate that.
5. You must use the information only in the document to make your decision.

Document:
{DOCUMENT_PLACEHOLDER}

Statement:
{STATEMENT_PLACEHOLDER}"""

CLEARCHECK_DIRECT = f"""\
Determine whether the provided statement is attributable to the document.
1. If any information in the statement is missing in the document, you should say "Not Attributable".
2. If any information in the statement is contradictory to the document, you should say "Contradictory".
3. If all information in the statement is supported by the document, you should say "Attributable".
4. You must use the information only in the document to make your decision.

Document:
{DOCUMENT_PLACEHOLDER}

Statement:
{STATEMENT_PLACEHOLDER}"""