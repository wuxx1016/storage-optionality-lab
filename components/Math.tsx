import katex from "katex";

type EquationProps = {
  children: string;
  label?: string;
  note?: string;
};

export function Equation({ children, label, note }: EquationProps) {
  const html = katex.renderToString(children, {
    displayMode: true,
    throwOnError: false,
    strict: false,
  });

  return (
    <figure className="equation-shell">
      <div className="equation-row">
        <div
          className="equation-scroll"
          dangerouslySetInnerHTML={{ __html: html }}
        />
        {label ? <span className="equation-label">({label})</span> : null}
      </div>
      {note ? <figcaption>{note}</figcaption> : null}
    </figure>
  );
}

type InlineMathProps = {
  children: string;
};

export function InlineMath({ children }: InlineMathProps) {
  const html = katex.renderToString(children, {
    displayMode: false,
    throwOnError: false,
    strict: false,
  });

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
