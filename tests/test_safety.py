import pytest
from red_curtain.safety import check_snapshot


def test_overheated_or_invalid_sensor_prevents_compute():
    card={'Temperature (Sensor edge) (C)':'37','Temperature (Sensor junction) (C)':'45',
          'Temperature (Sensor memory) (C)':'60'}
    check_snapshot({'card0':card})
    card['Temperature (Sensor memory) (C)']='91'
    with pytest.raises(RuntimeError,match='exceeds'):
        check_snapshot({'card0':card})
    del card['Temperature (Sensor memory) (C)']
    check_snapshot({'card0':card})
    card['Temperature (Sensor memory) (C)']='nan'
    with pytest.raises(RuntimeError,match='invalid'):
        check_snapshot({'card0':card})


@pytest.mark.parametrize('sensor',['edge','junction','memory','temp1'])
def test_one_valid_sensor_is_enough_on_cards_with_fewer_sensors(sensor):
    check_snapshot({'selected':{f'Temperature (Sensor {sensor}) (C)':'45',
                                'Temperature (Sensor memory) (C)':'N/A'}
                    if sensor!='memory' else {f'Temperature (Sensor {sensor}) (C)':'45'}})


@pytest.mark.parametrize('card',[{}, {'Temperature (Sensor edge) (C)':'N/A'},
                                {'Temperature (Sensor memory) (C)':None}])
def test_no_readable_sensor_refuses_compute(card):
    with pytest.raises(RuntimeError,match='No usable'):
        check_snapshot({'selected':card})


@pytest.mark.parametrize('reading',['nan','inf','-50','201','broken'])
def test_invalid_readings_never_become_zero(reading):
    with pytest.raises(RuntimeError,match='invalid'):
        check_snapshot({'selected':{'Temperature (Sensor edge) (C)':reading}})


def test_driver_critical_limit_is_respected_with_five_degree_margin():
    card={'Temperature (Sensor junction) (C)':'75',
          'Temperature critical (Sensor junction) (C)':'80'}
    with pytest.raises(RuntimeError,match='exceeds'):
        check_snapshot({'selected':card})
    card['Temperature (Sensor junction) (C)']='74'
    check_snapshot({'selected':card})
