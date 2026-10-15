from streamlit.testing.v1 import AppTest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'


def widget(app, kind, label):
    return next(w for w in getattr(app,kind) if w.label==label)


def test_report_generation_persistence_and_validation():
    app=AppTest.from_file(APP,default_timeout=60).run()
    assert not app.exception
    widget(app,'button','Generate technoeconomic report').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['study']['results'])==3
    assert any(m.label=='Base NPV' for m in app.metric)
    saved=app.session_state['study']['fingerprint']
    widget(app,'selectbox','Representative month').set_value('Jul').run()
    assert not app.exception
    assert app.session_state['study']['fingerprint']==saved
    widget(app,'number_input','Nameplate energy (kWh)').set_value(2500.).run()
    assert any('Inputs have changed' in w.value for w in app.warning)
    widget(app,'button','Generate technoeconomic report').click().run()
    assert not app.exception and not app.error
    assert app.session_state['study']['fingerprint']!=saved
    widget(app,'number_input','Reserve floor (%)').set_value(95.).run()
    widget(app,'button','Generate technoeconomic report').click().run()
    assert not app.exception
    assert any('SOC' in e.value for e in app.error)


def test_no_payback_zero_weekend_and_missing_csv_are_handled():
    app=AppTest.from_file(APP,default_timeout=60).run()
    widget(app,'number_input','Analysis period (years)').set_value(1).run()
    widget(app,'button','Generate technoeconomic report').click().run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label=='Simple payback')=='Not reached'
    widget(app,'slider','Weekend load relative to weekday (%)').set_value(0).run()
    widget(app,'button','Generate technoeconomic report').click().run()
    assert not app.exception and not app.error
    widget(app,'radio','Load input').set_value('24-hour CSV').run()
    assert widget(app,'button','Generate technoeconomic report').disabled
    assert app.session_state['study'] is not None
