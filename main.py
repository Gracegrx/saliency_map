def main():
    process_data()
    '''
    dist_options = ["dist", "visualise", "siamese", "saliency"]
    parser = argparse.ArgumentParser(description='delta s.')
    parser.add_argument('ds_type', type=str, help='options: {}'.format(dist_options))

    args = parser.parse_args()

    ds_type = args.ds_type
    if ds_type not in dist_options:
        return
    run(ds_type)
    '''


if __name__ == "__main__":
    main()
