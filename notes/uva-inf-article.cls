% LaTeX klasse voor artikelen en generieke documenten
% Onderdeel van het LaTeX template pakket UvA LaTeX
%
% Voor readme en oorspronkelijke  versie, zie
% https://gitlab-fnwi.uva.nl/informatica/LaTeX-template.git
%
% Geschreven door Robert van Wijk, met hulp van Frederick Kreuk, 
% Stephan van Schaik, Willem Vermeulen en Stephen Swatman.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uva-inf-article}
\LoadClass[10pt]{article}

\RequirePackage{hyperref}
\RequirePackage{verbatim}
\RequirePackage{attachfile2}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\RequirePackage{pgffor}
\RequirePackage{arrayjobx}
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage[document]{ragged2e}
\RequirePackage{lipsum}
\RequirePackage{translations}
\RequirePackage{ifthen}
\RequirePackage{xstring}
\RequirePackage[a4paper, left=30mm, right=30mm, top=20mm, bottom=30mm, marginpar=25mm]{geometry}
\RequirePackage[utf8]{inputenc}
\RequirePackage{csquotes}

\RequirePackage{tikz}

% Zodat we sinuswaarden kunnen berekenen voor de transparantie van de kruisjes.
\RequirePackage{xfp}


%------------------------------------------------------------------------------
%	COMMON
%------------------------------------------------------------------------------

\DeclareTranslation{dutch}{tutor}{Tutor}
\DeclareTranslation{english}{tutor}{Daily supervisor}

\DeclareTranslation{dutch}{of}{van}
\DeclareTranslation{english}{of}{of}

\DeclareTranslation{dutch}{lecturer}{Docent}
\DeclareTranslation{english}{lecturer}{Examiner}

\DeclareTranslation{dutch}{group}{Practicumgroep}
\DeclareTranslation{english}{group}{Group}

\DeclareTranslation{dutch}{student}{Student}
\DeclareTranslation{english}{student}{Student}

\DeclareTranslation{dutch}{students}{Studenten}
\DeclareTranslation{english}{students}{Students}

\DeclareTranslation{dutch}{course}{Cursus}
\DeclareTranslation{english}{course}{Course}

\DeclareTranslation{dutch}{course_id}{Vakcode}
\DeclareTranslation{english}{course_id}{Course code}

\DeclareTranslation{dutch}{words}{woorden}
\DeclareTranslation{english}{words}{words}

\DeclareTranslation{dutch}{uva_logo}{logo/logoUvA_nl}
\DeclareTranslation{english}{uva_logo}{logo/logoUvA_en}

\newcommand*{\assignment}[1]{\gdef\@assignment{#1}}
\newcommand*{\@assignment}{$\backslash assignment\{\}$ is unset}

\newcommand*{\assignmenttype}[1]{\gdef\@assignmenttype{#1}}
\newcommand*{\@assignmenttype}{$\backslash assignmenttype\{\}$ is unset}

\newcommand*{\tutor}[1]{\gdef\@tutor{#1}}
\newcommand*{\@tutor}{}

\newcommand*{\mentor}[1]{\gdef\@mentor{#1}}
\newcommand*{\@mentor}{}

\newcommand*{\group}[1]{\gdef\@group{#1}}
\newcommand*{\@group}{}

\newcommand*{\docent}[1]{\gdef\@docent{#1}}
\newcommand*{\@docent}{}

\newcommand*{\course}[1]{\gdef\@course{#1}}
\newcommand*{\@course}{}

\newcommand*{\courseid}[1]{\gdef\@courseid{#1}}
\newcommand*{\@courseid}{}

\newcounter{authorcount}
\setcounter{authorcount}{0}

\newcounter{uvacount}
\setcounter{uvacount}{0}

\newcommand\setprefix[2]{\expandafter\def\csname#1\endcsname{#2}}
\newcommand\getprefix[1]{\csname#1\endcsname}

\NewDocumentCommand\authors{>{\SplitList{;}}m}{
    \ProcessList{#1}{ \addauthor }
}
\newcommand\addauthor[1]{
    \stepcounter{authorcount}
    \setprefix{author\arabic{authorcount}}{#1}
}
\renewcommand{\author}[1]{
	\addauthor{#1}
}
\NewDocumentCommand\uvanetids{>{\SplitList{;}}m}{
    \ProcessList{#1}{ \adduvanetid }
}
\newcommand\adduvanetid[1]{
    \stepcounter{uvacount}
    \setprefix{uva\arabic{uvacount}}{#1}
}
\newcommand\uvanetid[1]{
	\adduvanetid{#1}
}

% Teken een andreaskruis. Nu met transparantie voor additioneel profijt!
\newcommand{\andreas}[5][1.0]{
    \begin{scope}[shift={(#2,#3)},rotate=45,scale=#4,transparency group,opacity=#1]
        \fill[fill=#5] (0,0) -- (0,1) -- (3,1) -- (3,0);
        \fill[fill=#5] (1,-1) -- (2,-1) -- (2,2) -- (1,2);
    \end{scope}
}

\newcommand{\andreasline}{
    \centering\begin{tikzpicture}
        \newcommand\numcross{29}
        \foreach \n in {1,...,\the\numexpr\numcross\relax} {
            \andreas[\fpeval{0.9*sin((\n + 1)*pi/(\numcross+2))}]{0.5*\n}{0}{0.10}{black}
        }
    \end{tikzpicture}
}

\newcommand{\andreaslineslim}{
    \centering\begin{tikzpicture}[baseline]
        \newcommand\numcross{74}
        \foreach \n in {1,...,\the\numexpr\numcross\relax} {
            \andreas[0.75]{0.2*\n}{0}{0.04}{black}
        }
    \end{tikzpicture}
}


%------------------------------------------------------------------------------
%	WOORDEN TELLEN
%------------------------------------------------------------------------------
% Header, footer, inhoudsopgave, samenvatting, referentielijst en bijlagen worden 
% NIET meegeteld, titel en referenties in tekst (als 1 woord) wel.
% Meer informatie: https://www.overleaf.com/learn/how-to/Is_there_a_way_to_run_a_word_count_that_doesn't_include_LaTeX_commands%3F

\newcommand{\quickwordcount}[1]{%
  \immediate\write18{texcount -1 -sum=1,1,1,0,0,1,1. -merge -q -nobib #1.tex > #1-words.sum }%
  \input{#1-words.sum} {\GetTranslation{words}}%
}

\newcommand{\detailtexcount}[1]{%
  \immediate\write18{texcount -merge -sum=1,0,1,0,0,1,1. -q #1.tex > #1.wcdetail }%
  \verbatiminput{#1.wcdetail}%
}

%------------------------------------------------------------------------------
%	INSTELLINGEN
%------------------------------------------------------------------------------

\reversemarginpar

\attachfilesetup{color=0 0 1}

\setlength{\RaggedRightParindent}{2em}

%------------------------------------------------------------------------------
%	HEADERS, FOOTERS EN SECTIONS
%------------------------------------------------------------------------------

\pagestyle{fancy}
\setlength{\headheight}{27pt}
\renewcommand{\headrulewidth}{0pt}
\lhead{\includegraphics[width=5cm]{\GetTranslation{uva_logo}}}
\chead{\raisebox{-\baselineskip}{\andreaslineslim}}
\rhead{\footnotesize \textsc {\@assignmenttype\\ \@assignment}}
\lfoot
{
}
\cfoot{\raisebox{\baselineskip}{\andreaslineslim}}
\rfoot{\small \textsc {\GetTranslation{page} \thepage\ \GetTranslation{of} \pageref{LastPage}}}

\fancypagestyle{firststyle}
 {
  \fancyhf{}
   \renewcommand{\headrulewidth}{0pt}
   \lfoot{}
   \cfoot{\raisebox{\baselineskip}{\andreaslineslim}}
   \rfoot{\small \textsc {\GetTranslation{page} \thepage\ \GetTranslation{of} \pageref{LastPage}}}
 }

\def\addlines#1{%
  \IfSubStr{#1}{ }{%
  	\saveexpandmode
  	\expandarg
    \StrSubstitute{#1}{ }{\noexpand\\}}{#1}}
    \restoreexpandmode

\newcommand{\sectionauthor}[2][]{
\section[#2]{#2\protect\marginpar{\raggedleft\noindent\normalfont\footnotesize #1}}
}

\newcommand{\subsectionauthor}[2][]{
\subsection[#2]{#2\protect\marginpar{\raggedleft\noindent\normalfont\footnotesize #1}}
}

%------------------------------------------------------------------------------
%	VOORPAGINA
%------------------------------------------------------------------------------

\renewcommand*{\maketitle}{%
\thispagestyle{empty}
\begin{center}
\includegraphics[width=15cm]{\GetTranslation{uva_logo}}
\\[2cm]
	\textsc{\huge \@assignment}\\[0.5cm]
	{ \huge \bfseries \@title}\\[0.5cm]
	\andreasline \\[0.3cm]
			\emph{\ifnum\value{authorcount}>1\relax
                    \GetTranslation{students}%
                  \else
                    \GetTranslation{student}%
                  \fi}\\
			\foreach \index in {1, ..., \value{authorcount}} {%
                {\Large \getprefix{author\index}\\[4.5cm]}}
\includegraphics[width=.75\textwidth]{logo/clslogo.png}
\begin{figure}[b]
\includegraphics[width=\textwidth]{logo/logo-combi-vu-uva-nl_web.jpg}
\end{figure}

\end{center}
\clearpage
}



