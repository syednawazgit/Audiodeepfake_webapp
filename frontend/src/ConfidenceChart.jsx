import {
  Cell,
  Label,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

const BONAFIDE = '#22c55e'
const SPOOF = '#ef4444'

const tooltipStyle = {
  backgroundColor: '#1e1e28',
  border: '1px solid rgba(108, 99, 255, 0.35)',
  borderRadius: 8,
  color: '#e8e8ef',
}

function renderCenterLabel(value) {
  return function CenterLabelContent({ viewBox }) {
    if (!viewBox || typeof viewBox.cx !== 'number') return null
    const { cx, cy } = viewBox
    return (
      <g>
        <text x={cx} y={cy - 4} textAnchor="middle" dominantBaseline="middle" fill="#f0f0f5" fontSize={26} fontWeight={800}>
          {value}
        </text>
        <text x={cx} y={cy + 18} textAnchor="middle" dominantBaseline="middle" fill="#9898a8" fontSize={11} fontWeight={500}>
          prediction confidence
        </text>
      </g>
    )
  }
}

export default function ConfidenceChart({ pctReal, pctFake, centerConfidence }) {
  const data = [
    { name: 'Bonafide', value: pctReal, color: BONAFIDE },
    { name: 'Spoof', value: pctFake, color: SPOOF },
  ]

  return (
    <div className="confidence-chart" role="img" aria-label="Bonafide versus spoof probability split">
      <ResponsiveContainer width="100%" height={280}>
        <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="46%"
            innerRadius={72}
            outerRadius={108}
            paddingAngle={2}
            stroke="#0f0f13"
            strokeWidth={2}
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} />
            ))}
            {centerConfidence != null && centerConfidence !== '' && (
              <Label content={renderCenterLabel(centerConfidence)} position="center" />
            )}
          </Pie>
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(4)}%`, 'Share']}
            contentStyle={tooltipStyle}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.name ?? ''}
          />
          <Legend
            verticalAlign="bottom"
            formatter={(value) => <span style={{ color: '#c8c8d4' }}>{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
